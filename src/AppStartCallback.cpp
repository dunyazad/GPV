#include <Common.h>
#include <App/App.h>
#include <App/AppEventHandlers.h>
#include <App/USBHandler.h>

#include <Algorithm/KDTree.h>
#include <Algorithm/SVO.h>
#include <Algorithm/Octree.hpp>

#include <Algorithm/CustomPolyDataFilter.h>
#include <Algorithm/vtkMedianFilter.h>
#include <Algorithm/vtkQuantizingFilter.h>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

int pid = 0;
size_t size_0 = 0;
size_t size_45 = 0;
float transform_0[16];
float transform_45[16];
Eigen::Vector3f cameraPosition;
unsigned char image_0[400 * 480];
unsigned char image_45[400 * 480];
Eigen::Vector3f points_0[400 * 480];
Eigen::Vector3f points_45[400 * 480];
Eigen::AlignedBox3f aabb_0(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));
Eigen::AlignedBox3f aabb_45(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));
Eigen::AlignedBox3f gaabb_0(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));
Eigen::AlignedBox3f gaabb_45(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));
Eigen::AlignedBox3f taabb(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));
Eigen::AlignedBox3f lmax(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));

vector<Eigen::Vector3f> patchPoints_0;
vector<Eigen::Vector3f> patchPoints_45;
vector<Eigen::Vector3f> inputPoints;

vector<Eigen::Matrix4f> cameraTransforms;

void LoadPatch(int patchID, vtkRenderer* renderer)
{
	stringstream ss;
	ss << "C:\\Debug\\Patches\\patch_" << patchID << ".pat";

	ifstream ifs;
	ifs.open(ss.str(), ios::in | ios::binary);

	ifs.read((char*)&pid, sizeof(int));
	ifs.read((char*)&size_0, sizeof(size_t));
	ifs.read((char*)&size_45, sizeof(size_t));
	ifs.read((char*)&transform_0, sizeof(float) * 16);
	ifs.read((char*)&transform_45, sizeof(float) * 16);
	ifs.read((char*)&cameraPosition, sizeof(float) * 3);
	ifs.read((char*)&image_0, sizeof(unsigned char) * 400 * 480);
	ifs.read((char*)&image_45, sizeof(unsigned char) * 400 * 480);
	ifs.read((char*)&points_0, sizeof(Eigen::Vector3f) * size_0);
	ifs.read((char*)&points_45, sizeof(Eigen::Vector3f) * size_45);

	ifs.close();

	vtkNew<vtkPoints> points;
	//points->SetNumberOfPoints(size_0);

	Eigen::Matrix4f t0(transform_0);
	Eigen::Matrix4f t45(transform_45);

	aabb_0 = Eigen::AlignedBox3f(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));
	aabb_45 = Eigen::AlignedBox3f(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));
	gaabb_0 = Eigen::AlignedBox3f(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));
	gaabb_45 = Eigen::AlignedBox3f(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));

	int count = 0;

	patchPoints_0.clear();
	patchPoints_45.clear();

	for (size_t i = 0; i < size_0; i++)
	{
		auto& p = points_0[i];

		if ((p.x() > -1000 && p.y() > -1000 && p.z() > -1000) &&
			(p.x() < 1000 && p.y() < 1000 && p.z() < 1000))
		{
			patchPoints_0.push_back(p);

			Eigen::Vector4f p4(p.x(), p.y(), p.z(), 1.0f);
			Eigen::Vector4f tp = t0 * p4;
			Eigen::Vector3f tp3 = Eigen::Vector3f(tp.x(), tp.y(), tp.z());

			points->InsertNextPoint(tp3.data());

			aabb_0.extend(p);
			gaabb_0.extend(tp3);
			taabb.extend(tp3);

			lmax.extend(aabb_0.max() - aabb_0.min());

			inputPoints.push_back(tp3);

			count++;
		}
	}

	for (size_t i = 0; i < size_45; i++)
	{
		auto& p = points_45[i];

		if ((p.x() > -1000 && p.y() > -1000 && p.z() > -1000) &&
			(p.x() < 1000 && p.y() < 1000 && p.z() < 1000))
		{
			patchPoints_45.push_back(p);

			Eigen::Vector4f p4(p.x(), p.y(), p.z(), 1.0f);
			Eigen::Vector4f tp = t45 * p4;
			Eigen::Vector3f tp3 = Eigen::Vector3f(tp.x(), tp.y(), tp.z());

			points->InsertNextPoint(tp3.data());

			aabb_45.extend(p);
			gaabb_45.extend(tp3);
			taabb.extend(tp3);

			lmax.extend(aabb_45.max() - aabb_45.min());

			inputPoints.push_back(tp3);

			count++;
		}
	}

	std::cout << aabb_0.min().transpose() << std::endl;
	std::cout << aabb_0.max().transpose() << std::endl;
	std::cout << gaabb_0.min().transpose() << std::endl;
	std::cout << gaabb_0.max().transpose() << std::endl;
	std::cout << aabb_45.min().transpose() << std::endl;
	std::cout << aabb_45.max().transpose() << std::endl;
	std::cout << gaabb_45.min().transpose() << std::endl;
	std::cout << gaabb_45.max().transpose() << std::endl;

	return;

	//VisualDebugging::AddCube("aabb", (aabb.min() + aabb.max()) * 0.5f, aabb.max() - aabb.min(), { 0.0f, 0.0f, 0.0f }, Color4::Red);

	vtkNew<vtkPolyData> polyData;
	polyData->SetPoints(points);

	//WritePLY(polyData, "C:\\Debug\\GPV\\Original.ply");

	//double spatialSigma = 0.5;  // adjust this based on the point cloud scale
	//double featureSigma = 0.1;  // adjust based on feature variance
	//double neighborhoodSize = 0.5;  // adjust based on the density of the point cloud

	//// Apply bilateral filter
	//vtkSmartPointer<vtkPoints> newPoints = BilateralFilter(polyData, spatialSigma, featureSigma, neighborhoodSize);

	//vtkNew<vtkPolyData> newPolyData;
	//newPolyData->SetPoints(newPoints);

	//WritePLY(newPolyData, "C:\\Debug\\GPV\\Filtered.ply");

	vtkNew<vtkVertexGlyphFilter> vertexFilter;
	vertexFilter->SetInputData(polyData);
	vertexFilter->Update();

	vtkNew<vtkPolyDataMapper> mapper;
	mapper->SetInputData(vertexFilter->GetOutput());

	vtkNew<vtkActor> actor;
	//actor->SetObjectName("points");
	actor->SetMapper(mapper);

	actor->GetProperty()->SetPointSize(5.0f);
	actor->GetProperty()->SetColor(1.0, 1.0, 1.0);

	renderer->AddActor(actor);
}

tuple<Eigen::Matrix4f, Eigen::Vector3f> LoadPatchTransform(int patchID)
{
	stringstream ss;
	ss << "C:\\Debug\\Patches\\patch_" << patchID << ".pat";

	ifstream ifs;
	ifs.open(ss.str(), ios::in | ios::binary);

	ifs.read((char*)&pid, sizeof(int));
	ifs.read((char*)&size_0, sizeof(size_t));
	ifs.read((char*)&size_45, sizeof(size_t));
	ifs.read((char*)&transform_0, sizeof(float) * 16);
	ifs.read((char*)&cameraPosition, sizeof(float) * 3);

	ifs.close();

	return make_tuple(Eigen::Matrix4f(transform_0), Eigen::Vector3f(cameraPosition));
}

void SaveTRNFile()
{
	ofstream ofs;
	ofs.open("C:\\Debug\\Patches\\transforms.trn", ios::out | ios::binary);

	int numberOfTransforms = 4252;
	ofs.write((char*)&numberOfTransforms, sizeof(int));

	for (size_t i = 0; i < 4252; i++)
	{
		printf("Patch : %4d\n", i);

		auto [transform, cameraPosition] = LoadPatchTransform(i);

		ofs.write((char*)transform.data(), sizeof(float) * 16);
		ofs.write((char*)cameraPosition.data(), sizeof(float) * 3);
	}
	ofs.close();
}

void LoadTRNFile()
{
	ifstream ifs;
	ifs.open("C:\\Debug\\Patches\\transforms.trn", ios::in | ios::binary);

	int numberOfTransforms = 0;
	ifs.read((char*)&numberOfTransforms, sizeof(int));

	for (size_t i = 0; i < numberOfTransforms; i++)
	{
		printf("Patch %4d\n", i);

		ifs.read((char*)&transform_0, sizeof(float) * 16);
		ifs.read((char*)&cameraPosition, sizeof(float) * 3);
		Eigen::Matrix4f transform(transform_0);

		cameraTransforms.push_back(transform);

		Eigen::Vector3f zero = (transform * Eigen::Vector4f(0.0f, 0.0f, 20.0f, 1.0f)).head<3>();
		Eigen::Vector3f right = (transform * Eigen::Vector4f(1.0f, 0.0f, 0.0f, 0.0f)).head<3>();
		Eigen::Vector3f up = (transform * Eigen::Vector4f(0.0f, 1.0f, 0.0f, 0.0f)).head<3>();
		Eigen::Vector3f front = (transform * Eigen::Vector4f(0.0f, 0.0f, -1.0f, 0.0f)).head<3>();
		Eigen::Vector3f cam = (transform * Eigen::Vector4f(cameraPosition.x(), cameraPosition.y(), cameraPosition.z(), 1.0f)).head<3>();

		//VisualDebugging::AddSphere("sphere", zero, { 0.5f, 0.5f, 0.5f }, { 0.0f, 0.0f, 1.0f }, Color4::Red);
		//VisualDebugging::AddLine("transform", zero, zero + right, Color4::Red);
		//VisualDebugging::AddLine("transform", zero, zero + up, Color4::Green);
		//VisualDebugging::AddLine("transform", zero, zero + (front * 10.0f), Color4::Yellow);
	}
}

void LoadModel(vtkRenderer* renderer, const string& filename)
{
	vtkNew<vtkPLYReader> reader;
	reader->SetFileName(filename.c_str());
	reader->Update();

	vtkPolyData* polyData = reader->GetOutput();

	vtkNew<vtkPolyDataMapper> mapper;
	mapper->SetInputData(polyData);

	vtkNew<vtkActor> actor;
	actor->SetMapper(mapper);

	renderer->AddActor(actor);

	renderer->ResetCamera();

	return;

	//vtkPoints* points = polyData->GetPoints();
	//vtkFloatArray* floatArray = vtkArrayDownCast<vtkFloatArray>(points->GetData());
	//float* pointData = static_cast<float*>(floatArray->GetPointer(0));
	//size_t numPoints = points->GetNumberOfPoints();

	//vtkCellArray* cells = polyData->GetPolys();
	//vtkIdType npts;
	//const vtkIdType* pts;

	//std::vector<uint32_t> triangleIndices;

	//cells->InitTraversal();
	//while (cells->GetNextCell(npts, pts))
	//{
	//	if (npts == 3)
	//	{
	//		for (vtkIdType i = 0; i < 3; ++i)
	//		{
	//			triangleIndices.push_back(static_cast<uint32_t>(pts[i]));
	//		}
	//	}
	//}

	//CUDA::Mesh mesh = CUDA::AllocateMesh(pointData, numPoints, triangleIndices.data(), triangleIndices.size() / 3);
	//CUDA::DeallocMesh(&mesh);

	//vector<Algorithm::kdNode> kdNodes(numPoints);
	//for (size_t i = 0; i < numPoints; i++)
	//{
	//	kdNodes[i].id = i;
	//	kdNodes[i].x[0] = pointData[i * 3 + 0];
	//	kdNodes[i].x[1] = pointData[i * 3 + 1];
	//	kdNodes[i].x[2] = pointData[i * 3 + 2];
	//}
	//Algorithm::kdTree tree;
	//tree.init(numPoints); // Allocates memory and prepares the tree

	//auto t = Time::Now();
	//// Build the KD-tree
	//tree.kdRoot = tree.buildTree(kdNodes.data(), numPoints, 0, 3);
	//t = Time::End(t, "KDTree Build");
}

int transformIndex = 0;
void MoveCamera(App* pApp, vtkCamera* camera, const Eigen::Matrix4f& tm)
{
	Eigen::Vector3f forward = tm.block<3, 1>(0, 2);
	Eigen::Vector3f position = tm.block<3, 1>(0, 3);

	Eigen::Vector3f cameraPosition = position + forward * 20.0f;

	camera->SetPosition(cameraPosition.x(), cameraPosition.y(), cameraPosition.z());
	camera->SetFocalPoint(position.x(), position.y(), position.z());
	//camera->SetParallelScale(10.0);

	//float pixel_to_world_ratio = 0.1;
	//float world_height = 480.0f * pixel_to_world_ratio;
	//camera->SetParallelScale(world_height / 2);

	camera->Modified();

	pApp->GetRenderer()->ResetCameraClippingRange();
	pApp->GetRenderWindow()->Render();

}

void CaptureNextFrame(App* pApp)
{
	if (transformIndex >= cameraTransforms.size()) return;

	vtkCamera* camera = pApp->GetRenderer()->GetActiveCamera();

	auto& tm = cameraTransforms[transformIndex];

	MoveCamera(pApp, camera, tm);

	//pApp->CaptureColorAndDepth("C:\\Resources\\2D\\Captured\\RGBD");
	pApp->CaptureAsPointCloud("C:\\Resources\\2D\\Captured\\PointCloud");
	printf("Saved %d\n", transformIndex);

	transformIndex++;
}

void LoadDepthImage()
{
	std::string depthmapFileName = "C:\\Resources\\2D\\Captured\\RGBD\\depth_0.png";

	vtkSmartPointer<vtkPNGReader> reader = vtkSmartPointer<vtkPNGReader>::New();
	reader->SetFileName(depthmapFileName.c_str());
	reader->Update();

	vtkImageData* imageData = reader->GetOutput();
	int* dims = imageData->GetDimensions();

	vtkUnsignedCharArray* pixelArray = vtkUnsignedCharArray::SafeDownCast(imageData->GetPointData()->GetScalars());

	if (!pixelArray) {
		std::cerr << "Failed to get pixel data from image!" << std::endl;
		return;
	}

	for (int z = 0; z < dims[2]; z++) {
		for (int y = 0; y < dims[1]; y++) {
			for (int x = 0; x < dims[0]; x++) {
				int idx = z * dims[0] * dims[1] + y * dims[0] + x;
				unsigned char pixelValue = pixelArray->GetValue(idx);

				double depthValue = static_cast<double>(pixelValue) / 255.0;

				depthValue *= 200;

				// Print out depth value if needed
				//std::cout << "Depth value at (" << x << ", " << y << "): " << depthValue << std::endl;

				Eigen::Vector4f p4(x * 0.05f - 12.8f, y * 0.05f - 24.0f, -depthValue, 1.0f);
				Eigen::Vector3f p = (cameraTransforms[0] * p4).head<3>();

				VisualDebugging::AddSphere("depth", p, { 0.05f, 0.05f, 0.05f }, { 0.0f, 0.0f, 1.0f }, Color4::Red);
			}
		}
	}
}

bool enabledToCapture = false;

void AppStartCallback_Capture(App* pApp)
{
	pApp->AddAppUpdateCallback([&](App* pApp) {
		if (enabledToCapture)
		{
			CaptureNextFrame(pApp);
		}

		static size_t index = 0;

		auto kdTreePoints = (vector<Eigen::Vector3f>*)pApp->registry["kdTreePoints"];
		auto kdTreeColors = (vector<Color4>*)pApp->registry["kdTreeColors"];

		auto& p = (*kdTreePoints)[index];
		auto& c = (*kdTreeColors)[index];

		VD::AddSphere("points",
			p,
			{ 0.05f,0.05f,0.05f },
			{ 0.0f, 0.0f, 1.0f },
			c);
		index++;
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
	// Parallel Scale�� ī�޶� ���� ����
	// �ȼ��� 3D ������ ���� * â ���� / 2
	// ���⿡�� 256 x 480�̹Ƿ� �ȼ��� 0.1, â���� 480
	// 480 * 0.1 / 2 = 24
	camera->SetParallelScale(24);

	//SaveTRNFile();

	LoadTRNFile();

	//LoadDepthImage();

	//VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 100.0f, 0.0f, 0.0f }, Color4::Red);
	//VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 100.0f, 0.0f }, Color4::Green);
	//VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, 100.0f }, Color4::Blue);

	enabledToCapture = true;
}

struct Point
{
	Eigen::Vector3f position;
	Eigen::Vector3f normal;
	Eigen::Vector3f color;
};

void AppStartCallback_Convert(App* pApp)
{
	auto renderer = pApp->GetRenderer();
	Point* points = new Point[256 * 480];
	size_t numberOfPoints = 0;

	for (size_t i = 0; i < 4252; i++)
	{
		auto te = Time::Now();

		numberOfPoints = 0;
		memset(points, 0, sizeof(Point) * 256 * 480);

		stringstream ss;
		ss << "C:\\Resources\\2D\\Captured\\PointCloud\\point_" << i << ".ply";

		vtkNew<vtkPLYReader> reader;
		reader->SetFileName(ss.str().c_str());
		reader->Update();

		vtkPolyData* polyData = reader->GetOutput();

		auto plyPoints = polyData->GetPoints();
		vtkDataArray* plyNormals = polyData->GetPointData()->GetNormals();
		vtkUnsignedCharArray* plyColors = vtkUnsignedCharArray::SafeDownCast(polyData->GetPointData()->GetScalars());

		for (size_t pi = 0; pi < plyPoints->GetNumberOfPoints(); pi++)
		{
			auto dp = plyPoints->GetPoint(pi);
			auto normal = plyNormals->GetTuple(pi);
			unsigned char color[3];
			plyColors->GetTypedTuple(pi, color);

			points[pi].position.x() = dp[0];
			points[pi].position.y() = dp[1];
			points[pi].position.z() = dp[2];

			points[pi].normal.x() = normal[0];
			points[pi].normal.y() = normal[1];
			points[pi].normal.z() = normal[2];

			points[pi].color.x() = (float)color[0] / 255.0f;
			points[pi].color.y() = (float)color[1] / 255.0f;
			points[pi].color.z() = (float)color[2] / 255.0f;

			numberOfPoints++;
		}

		stringstream oss;
		oss << "C:\\Debug\\Patches\\point_" << i << ".pnt";
		ofstream ofs;
		ofs.open(oss.str(), ios::out | ios::binary);

		ofs.write((char*)&numberOfPoints, sizeof(size_t));
		ofs.write((char*)points, numberOfPoints * sizeof(Point));
		ofs.close();

		Time::End(te, "Convering PointCloud Patch", i);
	}

	delete[] points;
}

void AppStartCallback_LoadPNT(App* pApp)
{
	auto renderer = pApp->GetRenderer();
	Point* points = new Point[256 * 480];
	size_t numberOfPoints = 0;

	float voxelSize = 0.1f;
	int volumeDimensionX = 1000;
	int volumeDimensionY = 1000;
	int volumeDimensionZ = 500;

	LoadModel(renderer, "C:\\Resources\\3D\\PLY\\Complete\\Lower.ply");

	//SaveTRNFile();
	LoadTRNFile();

	Eigen::Vector3f modelTranslation(20.0f, 75.0f, 25.0f);

	//CUDA::Voxel* volume;
	//cudaMallocManaged(&volume, sizeof(CUDA::Voxel) * volumeDimensionX * volumeDimensionY * volumeDimensionZ);
	//cudaDeviceSynchronize();

	//Eigen::Vector3f* inputPoints = nullptr;
	//cudaMallocManaged(&inputPoints, sizeof(Eigen::Vector3f) * 256 * 480);

	//for (size_t i = 0; i < 4252; i++)
	//{
	//	auto te = Time::Now();

	//	memset(points, 0, sizeof(Point) * 256 * 480);

	//	stringstream ss;
	//	ss << "C:\\Debug\\Patches\\point_" << i << ".pnt";
	//	ifstream ifs;
	//	ifs.open(ss.str(), ios::in | ios::binary);
	//	ifs.read((char*)&numberOfPoints, sizeof(size_t));
	//	ifs.read((char*)points, numberOfPoints * sizeof(Point));
	//	ifs.close();

	//	for (size_t pi = 0; pi < numberOfPoints; pi++)
	//	{
	//		auto& p = points[pi].position;
	//		auto& n = points[pi].normal;
	//		auto& c = points[pi].color;
	//		p += modelTranslation;

	//		inputPoints[pi] = p;
	//	}

	//	CUDA::IntegrateInputPoints(
	//		volume,
	//		make_int3(volumeDimensionX, volumeDimensionY, volumeDimensionZ),
	//		0.1f,
	//		inputPoints,
	//		numberOfPoints);

	//	Time::End(te, "Loading PointCloud Patch", i);
	//}
	//delete[] points;

	//auto t = Time::Now();
	//for (size_t i = 0; i < volumeDimensionX * volumeDimensionY * volumeDimensionZ; i++)
	//{
	//	if (volume[i].tsdfValue != 1.0f) continue;

	//	int zKey = i / (volumeDimensionX * volumeDimensionY);
	//	int yKey = (i % (volumeDimensionX * volumeDimensionY)) / volumeDimensionX;
	//	int xKey = (i % (volumeDimensionX * volumeDimensionY)) % volumeDimensionX;

	//	float x = xKey * voxelSize - modelTranslation.x();
	//	float y = yKey * voxelSize - modelTranslation.y();
	//	float z = zKey * voxelSize - modelTranslation.z();

	//	//printf("%f, %f, %f\n", x, y, z);

	//	VD::AddCube("temp", { x, y, z }, 0.1f, Color4::Red);
	//}
	//Time::End(t, "Show Voxels");

	//cudaFree(volume);

	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 100.0f, 0.0f, 0.0f }, Color4::Red);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 100.0f, 0.0f }, Color4::Green);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, 100.0f }, Color4::Blue);
}

void AppStartCallback_Integrate(App* pApp)
{
	//auto renderer = pApp->GetRenderer();

	//float voxelSize = 0.1f;
	//int volumeDimensionX = 1000;
	//int volumeDimensionY = 1000;
	//int volumeDimensionZ = 500;

	//LoadModel(renderer, "C:\\Resources\\3D\\PLY\\Complete\\Lower.ply");

	////SaveTRNFile();
	//LoadTRNFile();

	//Eigen::Vector3f modelTranslation(20.0f, 75.0f, 25.0f);

	//CUDA::Voxel* volume;
	//cudaMallocManaged(&volume, sizeof(CUDA::Voxel) * volumeDimensionX * volumeDimensionY * volumeDimensionZ);
	//cudaDeviceSynchronize();

	//Eigen::Vector3f* inputPoints = nullptr;
	//cudaMallocManaged(&inputPoints, sizeof(Eigen::Vector3f) * 256 * 480);

	//size_t numberOfInputPoints = 0;

	///*
	//{
	//	float voxelSize = 0.1f;
	//	int volumeDimensionX = 1000;
	//	int volumeDimensionY = 1000;
	//	int volumeDimensionZ = 1000;

	//	auto t = Time::Now();

	//	Eigen::AlignedBox3f aabb;
	//	vector<Eigen::Vector3f> loadedPoints;
	//	int loadedCount = 0;

	//	//size_t i = 3;
	//	//for (size_t i = 400; i < 500; i++)
	//	for (size_t i = 0; i < 4252; i++)
	//	{
	//		auto te = Time::Now();

	//		stringstream ss;
	//		ss << "C:\\Resources\\2D\\Captured\\PointCloud\\point_" << i << ".ply";

	//		vtkNew<vtkPLYReader> reader;
	//		reader->SetFileName(ss.str().c_str());
	//		reader->Update();

	//		vtkPolyData* polyData = reader->GetOutput();

	//		auto points = polyData->GetPoints();
	//		for (size_t pi = 0; pi < points->GetNumberOfPoints(); pi++)
	//		{
	//			auto dp = points->GetPoint(pi);
	//			//Eigen::Vector4f tp = cameraTransforms[i] * Eigen::Vector4f(dp[0], dp[1], dp[2] + 20.0f, 1.0f);
	//			//Eigen::Vector3f p(tp.x(), tp.y(), tp.z());

	//			Eigen::Vector3f p(dp[0], dp[1], dp[2]);
	//			p += modelTranslation;

	//			int xKey = (int)floorf(p.x() / voxelSize);
	//			int yKey = (int)floorf(p.y() / voxelSize);
	//			int zKey = (int)floorf(p.z() / voxelSize);

	//			if (xKey < 0 || xKey >= volumeDimensionX) continue;
	//			if (yKey < 0 || yKey >= volumeDimensionY) continue;
	//			if (zKey < 0 || zKey >= volumeDimensionZ) continue;

	//			volume[zKey * volumeDimensionX * volumeDimensionY + yKey * volumeDimensionX + xKey].tsdfValue = 1.0f;
	//			//VD::AddCube("temp", p, 0.1f, Color4::Red);
	//		}

	//		Time::End(te, "Loading PointCloud Patch", i);
	//	}

	//	t = Time::Now();
	//	for (size_t i = 0; i < volumeDimensionX * volumeDimensionY * volumeDimensionZ; i++)
	//	{
	//		if (volume[i].tsdfValue != 1.0f) continue;

	//		int zKey = i / (volumeDimensionX * volumeDimensionY);
	//		int yKey = (i % (volumeDimensionX * volumeDimensionY)) / volumeDimensionX;
	//		int xKey = (i % (volumeDimensionX * volumeDimensionY)) % volumeDimensionX;

	//		float x = xKey * voxelSize - modelTranslation.x();
	//		float y = yKey * voxelSize - modelTranslation.y();
	//		float z = zKey * voxelSize - modelTranslation.z();

	//		//printf("%f, %f, %f\n", x, y, z);

	//		VD::AddCube("temp", {x, y, z}, 0.1f, Color4::Red);
	//	}
	//	Time::End(t, "Show Voxels");

	//	//for (auto& kvp : volume)
	//	//{
	//	//	auto p = GetPosition(kvp.first);
	//	//	VD::AddCube("voxels", p, { 0.1f, 0.1f, 0.1f }, { 0.0f, 0.0f, 1.0f }, Color4::Red);
	//	//}
	//}
	//*/

	//{
	//	auto t = Time::Now();

	//	Eigen::AlignedBox3f aabb;
	//	vector<Eigen::Vector3f> loadedPoints;
	//	int loadedCount = 0;

	//	//size_t i = 3;
	//	//for (size_t i = 0; i < 1000; i++)
	//	for (size_t i = 0; i < 4252; i++)
	//	{
	//		cudaMemset(inputPoints, 0.0f, sizeof(Eigen::Vector3f) * 256 * 480);
	//		numberOfInputPoints = 0;

	//		auto te = Time::Now();

	//		stringstream ss;
	//		ss << "C:\\Resources\\2D\\Captured\\PointCloud\\point_" << i << ".ply";

	//		vtkNew<vtkPLYReader> reader;
	//		reader->SetFileName(ss.str().c_str());
	//		reader->Update();

	//		vtkPolyData* polyData = reader->GetOutput();

	//		auto points = polyData->GetPoints();
	//		for (size_t pi = 0; pi < points->GetNumberOfPoints(); pi++)
	//		{
	//			auto dp = points->GetPoint(pi);
	//			//Eigen::Vector4f tp = cameraTransforms[i] * Eigen::Vector4f(dp[0], dp[1], dp[2] + 20.0f, 1.0f);
	//			//Eigen::Vector3f p(tp.x(), tp.y(), tp.z());

	//			Eigen::Vector3f p(dp[0], dp[1], dp[2]);
	//			p += modelTranslation;

	//			inputPoints[pi] = p;
	//			numberOfInputPoints++;
	//		}

	//		CUDA::IntegrateInputPoints(
	//			volume,
	//			make_int3(volumeDimensionX, volumeDimensionY, volumeDimensionZ),
	//			0.1f,
	//			inputPoints,
	//			numberOfInputPoints);

	//		Time::End(te, "Loading PointCloud Patch", i);
	//	}

	//	t = Time::Now();
	//	for (size_t i = 0; i < volumeDimensionX * volumeDimensionY * volumeDimensionZ; i++)
	//	{
	//		if (volume[i].tsdfValue != 1.0f) continue;

	//		int zKey = i / (volumeDimensionX * volumeDimensionY);
	//		int yKey = (i % (volumeDimensionX * volumeDimensionY)) / volumeDimensionX;
	//		int xKey = (i % (volumeDimensionX * volumeDimensionY)) % volumeDimensionX;

	//		float x = xKey * voxelSize - modelTranslation.x();
	//		float y = yKey * voxelSize - modelTranslation.y();
	//		float z = zKey * voxelSize - modelTranslation.z();

	//		//printf("%f, %f, %f\n", x, y, z);

	//		VD::AddCube("temp", { x, y, z }, 0.1f, Color4::Red);
	//	}
	//	Time::End(t, "Show Voxels");

	//	//for (auto& kvp : volume)
	//	//{
	//	//	auto p = GetPosition(kvp.first);
	//	//	VD::AddCube("voxels", p, { 0.1f, 0.1f, 0.1f }, { 0.0f, 0.0f, 1.0f }, Color4::Red);
	//	//}
	//}

	//cudaFree(volume);

	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 100.0f, 0.0f, 0.0f }, Color4::Red);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 100.0f, 0.0f }, Color4::Green);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, 100.0f }, Color4::Blue);
}

void AppStartCallback_KDTree(App* pApp)
{
	auto renderer = pApp->GetRenderer();
	//LoadModel(renderer, "C:\\Resources\\3D\\PLY\\Complete\\Lower.ply");

	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 100.0f, 0.0f, 0.0f }, Color4::Red);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 100.0f, 0.0f }, Color4::Green);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, 100.0f }, Color4::Blue);

	vtkNew<vtkPLYReader> reader;
	reader->SetFileName("C:\\Resources\\3D\\PLY\\Complete\\Lower_pointcloud.ply");
	reader->Update();

	vtkPolyData* polyData = reader->GetOutput();

	auto plyPoints = polyData->GetPoints();
	vtkDataArray* plyNormals = polyData->GetPointData()->GetNormals();
	vtkUnsignedCharArray* plyColors = vtkUnsignedCharArray::SafeDownCast(polyData->GetPointData()->GetScalars());

	static vector<Eigen::Vector3f> points;
	static vector<Color4> colors;

	vector<unsigned int> pointIndices;

	for (size_t pi = 0; pi < plyPoints->GetNumberOfPoints(); pi++)
	{
		pointIndices.push_back((unsigned int)pi);

		auto dp = plyPoints->GetPoint(pi);
		auto normal = plyNormals->GetTuple(pi);
		unsigned char color[3];
		plyColors->GetTypedTuple(pi, color);

		points.push_back({ (float)dp[0], (float)dp[1], (float)dp[2] });
		colors.push_back(Color4(color[0], color[1], color[2], 255));

		//VD::AddSphere("points",
		//	{ (float)dp[0], (float)dp[1], (float)dp[2] },
		//	{ 0.05f,0.05f,0.05f },
		//	{ 0.0f, 0.0f, 1.0f },
		//	Color4(color[0], color[1], color[2], 255));
	}

	auto t = Time::Now();

	Algorithm::KDTree* kdtree = new Algorithm::KDTree;
	kdtree->SetPoints((float*)points.data());
	kdtree->BuildTree(pointIndices);

	//for (size_t i = 0; i < plyPoints->GetNumberOfPoints(); i++)
	//{
	//	kdtree->Insert(i);
	//}

	pApp->registry["kdtree"] = kdtree;
	pApp->registry["points"] = &points;

	t = Time::End(t, "Building KDTree");

	size_t count = 0;
	kdtree->TraversePreOrder([&](Algorithm::KDTreeNode* node) {
		count++;
		auto i = node->GetPointIndex();
		auto& p = points[i];
		auto& c = colors[i];

		VD::AddSphere("points",
			p,
			{ 0.05f,0.05f,0.05f },
			{ 0.0f, 0.0f, 1.0f },
			c);
		});

	printf("count : %llu\n", count);
}

void AppStartCallback_Octree(App* pApp)
{
	auto renderer = pApp->GetRenderer();
	//LoadModel(renderer, "C:\\Resources\\3D\\PLY\\Complete\\Lower.ply");

	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 100.0f, 0.0f, 0.0f }, Color4::Red);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 100.0f, 0.0f }, Color4::Green);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, 100.0f }, Color4::Blue);

	vtkNew<vtkPLYReader> reader;
	//reader->SetFileName("C:\\Resources\\3D\\PLY\\Complete\\Lower_pointcloud.ply");
	reader->SetFileName("./../../res/3D/Lower_pointcloud.ply");
	reader->Update();

	vtkPolyData* polyData = reader->GetOutput();

	auto plyPoints = polyData->GetPoints();
	float* rawPoints = static_cast<float*>(plyPoints->GetData()->GetVoidPointer(0));
	vtkDataArray* plyNormals = polyData->GetPointData()->GetNormals();
	float* rawNormals = static_cast<float*>(plyNormals->GetVoidPointer(0));
	vtkUnsignedCharArray* plyColors = vtkUnsignedCharArray::SafeDownCast(polyData->GetPointData()->GetScalars());

	static vector<Eigen::Vector3f> points;
	static vector<Color4> colors;

	vector<unsigned int> pointIndices;

	for (size_t pi = 0; pi < plyPoints->GetNumberOfPoints(); pi++)
	{
		pointIndices.push_back((unsigned int)pi);

		auto dp = plyPoints->GetPoint(pi);
		auto normal = plyNormals->GetTuple(pi);
		unsigned char color[3];
		plyColors->GetTypedTuple(pi, color);

		points.push_back({ (float)dp[0], (float)dp[1], (float)dp[2] });
		colors.push_back(Color4(color[0], color[1], color[2], 255));

		VD::AddSphere("points",
			{ (float)dp[0], (float)dp[1], (float)dp[2] },
			{ 0.1f,0.1f,0.1f },
			{ 0.0f, 0.0f, 1.0f },
			Color4(color[0], color[1], color[2], 255));
	}

	auto t = Time::Now();

	auto bounds = polyData->GetBounds();
	Eigen::AlignedBox3f aabb(
		Eigen::Vector3f { (float)bounds[0], (float)bounds[2], (float)bounds[4] },
		Eigen::Vector3f { (float)bounds[1], (float)bounds[3], (float)bounds[5] });
	static Spatial::Octree octree(aabb, 8, 10000);

	pApp->registry["octree"] = &octree;

	Eigen::Vector3f* newPoints = new Eigen::Vector3f[plyPoints->GetNumberOfPoints()];
	memcpy(newPoints, rawPoints, sizeof(Eigen::Vector3f) * plyPoints->GetNumberOfPoints());
	Eigen::Vector3f* newNormals = new Eigen::Vector3f[plyPoints->GetNumberOfPoints()];
	memcpy(newNormals, rawNormals, sizeof(Eigen::Vector3f) * plyPoints->GetNumberOfPoints());
	pApp->registry["octree_points"] = newPoints;
	pApp->registry["octree_normals"] = newNormals;
	octree.setPoints(newPoints, newNormals, plyPoints->GetNumberOfPoints());

	for (size_t pi = 0; pi < plyPoints->GetNumberOfPoints(); pi++)
	{
		octree.insert(pi);
	}

	t = Time::End(t, "Octree Building");

	//octree.traverse([&](Spatial::OctreeNode* node) {
	//	if (node->children[0] == nullptr && node->pointIndices.size() != 0)
	//	{
	//		Eigen::Vector3f center = node->aabb.center();
	//		Eigen::Vector3f scale = node->aabb.max() - node->aabb.min();
	//		VD::AddCube("OctreeNodes", center, scale, {0.0f, 0.0f, 1.0f}, Color4::Red);
	//	}
	//	});
}

#pragma region Poisson
	//struct Voxel {
	//	float divergence = 0.0f;
	//	float scalarField = 0.0f;
	//	Eigen::Vector3f color = Eigen::Vector3f(1.0f, 1.0f, 1.0f); // 초기화

	//	Voxel() :divergence(0.0f), scalarField(0.0f), color(Eigen::Vector3f(1.0f, 1.0f, 1.0f))
	//	{}
	//};

	//// Jacobi 반복법으로 스칼라 필드 계산
	//void solvePoissonEquation(vector<vector<vector<Voxel>>>&grid, int iterations) {
	//	int resolution = grid.size();
	//	float delta = 1.0f / resolution;

	//	for (int iter = 0; iter < iterations; ++iter) {
	//		// 새로운 스칼라 필드 저장
	//		vector<vector<vector<float>>> newScalarField(
	//			resolution, vector<vector<float>>(
	//				resolution, vector<float>(resolution, 0.0f)));

	//		// Jacobi 반복법
	//		for (int x = 1; x < resolution - 1; ++x) {
	//			for (int y = 1; y < resolution - 1; ++y) {
	//				for (int z = 1; z < resolution - 1; ++z) {
	//					float laplacian = (
	//						grid[x + 1][y][z].scalarField +
	//						grid[x - 1][y][z].scalarField +
	//						grid[x][y + 1][z].scalarField +
	//						grid[x][y - 1][z].scalarField +
	//						grid[x][y][z + 1].scalarField +
	//						grid[x][y][z - 1].scalarField -
	//						6.0f * grid[x][y][z].scalarField
	//						) / (delta * delta);

	//					// 포아송 방정식 갱신
	//					newScalarField[x][y][z] = (laplacian - grid[x][y][z].divergence) / 6.0f;
	//				}
	//			}
	//		}

	//		// 스칼라 필드 업데이트
	//		for (int x = 0; x < resolution; ++x) {
	//			for (int y = 0; y < resolution; ++y) {
	//				for (int z = 0; z < resolution; ++z) {
	//					grid[x][y][z].scalarField = newScalarField[x][y][z];
	//				}
	//			}
	//		}
	//	}
	//}

	//void AppStartCallback_Poisson(App * pApp)
	//{
	//	auto renderer = pApp->GetRenderer();
	//	//LoadModel(renderer, "C:\\Resources\\3D\\PLY\\Complete\\Lower.ply");

	//	//VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 100.0f, 0.0f, 0.0f }, Color4::Red);
	//	//VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 100.0f, 0.0f }, Color4::Green);
	//	//VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, 100.0f }, Color4::Blue);

	//	vtkNew<vtkPLYReader> reader;
	//	//reader->SetFileName("C:\\Resources\\3D\\PLY\\Complete\\Lower_pointcloud.ply");
	//	reader->SetFileName("./../../res/3D/Lower_pointcloud.ply");
	//	reader->Update();

	//	vtkPolyData* polyData = reader->GetOutput();

	//	auto plyPoints = polyData->GetPoints();
	//	float* rawPoints = static_cast<float*>(plyPoints->GetData()->GetVoidPointer(0));
	//	vtkDataArray* plyNormals = polyData->GetPointData()->GetNormals();
	//	float* rawNormals = static_cast<float*>(plyNormals->GetVoidPointer(0));
	//	vtkUnsignedCharArray* plyColors = vtkUnsignedCharArray::SafeDownCast(polyData->GetPointData()->GetScalars());

	//	static vector<Eigen::Vector3f> points;
	//	static vector<Eigen::Vector3f> normals;
	//	static vector<Color4> colors;

	//	vector<unsigned int> pointIndices;

	//	auto bounds = polyData->GetBounds();
	//	Eigen::AlignedBox3f aabb(
	//		Eigen::Vector3f{ (float)bounds[0], (float)bounds[2], (float)bounds[4] },
	//		Eigen::Vector3f{ (float)bounds[1], (float)bounds[3], (float)bounds[5] });
	//	Eigen::Vector3f center = aabb.center();

	//	Eigen::Vector3f aabbDelta = aabb.max() - aabb.min() - center;
	//	float axisMax = aabbDelta.x();
	//	if (axisMax < aabbDelta.y()) axisMax = aabbDelta.y();
	//	if (axisMax < aabbDelta.z()) axisMax = aabbDelta.z();

	//	struct Voxel
	//	{
	//		int count = 0;
	//		Eigen::Vector3f normal;
	//		Eigen::Vector3f color;
	//	};

	//	auto t = Time::Now();

	//	float voxelSize = 0.1f;
	//	vector<Voxel> volume;
	//	volume.resize(1000 * 1000 * 1000);

	//	t = Time::End(t, "Initialize volume");

	//	for (size_t pi = 0; pi < plyPoints->GetNumberOfPoints(); pi++)
	//	{
	//		pointIndices.push_back((unsigned int)pi);

	//		auto dp = plyPoints->GetPoint(pi);
	//		auto normal = plyNormals->GetTuple(pi);
	//		unsigned char color[3];
	//		plyColors->GetTypedTuple(pi, color);

	//		Eigen::Vector3f point((float)dp[0] - center.x(), (float)dp[1] - center.y(), (float)dp[2] - center.z());
	//		points.push_back(point);
	//		Eigen::Vector3f pointNormal((float)normal[0], (float)normal[1], (float)normal[2]);
	//		normals.push_back(pointNormal);
	//		auto color4 = Color4(color[0], color[1], color[2], 255);
	//		colors.push_back(color4);
	//		Eigen::Vector3f pointColor((float)color4.x() / 255.0f, (float)color4.y() / 255.0f, (float)color4.z() / 255.0f);

	//		//VD::AddSphere("points",
	//		//	point,
	//		//	{ 0.1f,0.1f,0.1f },
	//		//	{ 0.0f, 0.0f, 1.0f },
	//		//	Color4(color[0], color[1], color[2], 255));

	//		int xIndex = (int)floorf((point.x() + 50.0f) / voxelSize);
	//		int yIndex = (int)floorf((point.y() + 50.0f) / voxelSize);
	//		int zIndex = (int)floorf((point.z() + 50.0f) / voxelSize);
	//		size_t index = zIndex * 1000 * 1000 + yIndex * 1000 + xIndex;

	//		if (volume[index].count == 0)
	//		{
	//			volume[index].normal = pointNormal;
	//			volume[index].color = pointColor;
	//			volume[index].count++;
	//		}
	//		else
	//		{
	//			volume[index].normal += pointNormal;
	//			volume[index].color += pointColor;
	//			volume[index].count++;
	//		}
	//	}

	//	t = Time::End(t, "Integrate points");

	//	for (size_t i = 0; i < volume.size(); i++)
	//	{
	//		int zIndex = i / (1000 * 1000);
	//		int yIndex = (i % (1000 * 1000)) / 1000;
	//		int xIndex = (i % (1000 * 1000)) % 1000;

	//		if (0 < volume[i].count)
	//		{
	//			Eigen::Vector3f position((float)xIndex * voxelSize, (float)yIndex * voxelSize, (float)zIndex * voxelSize);

	//			Eigen::Vector3f c = volume[i].color.normalized();
	//			Color4 c4(255, 255, 255, 255);

	//			c4.FromNormalzed(c.x(), c.y(), c.z(), 1.0f);

	//			VD::AddCube("voxel", { (float)xIndex * voxelSize - 50.0f, (float)yIndex * voxelSize - 50.0f, (float)zIndex * voxelSize - 50.0f }, 0.05f, c4);
	//			//cout << position << endl;

	//			//VD::AddSphere("points",
	//			//	{ (float)xIndex * voxelSize, (float)yIndex * voxelSize, (float)zIndex * voxelSize },
	//			//	{ 0.1f,0.1f,0.1f },
	//			//	{ 0.0f, 0.0f, 1.0f },
	//			//	c4);
	//		}
	//	}

	//	//VD::AddCube("AABC", { 0.0f, 0.0f, 0.0f }, axisMax * 0.5f, Color4::White);
	//	//printf("axisMax : %f\n", axisMax);

	//	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { axisMax * 0.5f, 0.0f, 0.0f }, Color4::Red);
	//	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, axisMax * 0.5f, 0.0f }, Color4::Green);
	//	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, axisMax * 0.5f }, Color4::Blue);

	//	t = Time::End(t, "Visualize");
	//}
#pragma endregion

int edgeTable[256] = {
	0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
	0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
	0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
	0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
	0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
	0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
	0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
	0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
	0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
	0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
	0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
	0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
	0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
	0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
	0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
	0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
	0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
	0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
	0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
	0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
	0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
	0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
	0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
	0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
	0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
	0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
	0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
	0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
	0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
	0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
	0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
	0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0 };

int triTable[256][16] = {
{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
{3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
{3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
{3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
{9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
{9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
{8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
{9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
{3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
{4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
{9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
{5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
{9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
{0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
{10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
{5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
{9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
{0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
{1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
{2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
{7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
{11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
{11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
{1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
{9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
{2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
{0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
{6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
{6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
{5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
{1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
{6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
{3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
{10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
{10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
{1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
{0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
{10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
{3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
{6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
{10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
{7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
{7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
{0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
{7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
{10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
{2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
{7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
{2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
{10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
{7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
{6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
{8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
{6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
{8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
{0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
{1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
{10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
{10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
{5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
{9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
{7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
{6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
{6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
{9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
{1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
{0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
{5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
{11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
{2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
{1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
{9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
{9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
{9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
{5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
{8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
{0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
{9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
{11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
{1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
{4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
{0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
{3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
{0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
{9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
{1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1} };

struct Voxel
{
	int count = 0;
	float weight = 0.0f;
	float tsdfValue = FLT_MAX;

	Eigen::Vector3f normal;
	Eigen::Vector3f color;
};

struct Mesh
{
	vector<Eigen::Vector3f> vertices;
	vector<unsigned int> indices;
	vector<Eigen::Vector3f> normals;
	vector<Eigen::Vector3f> colors;
};

//Mesh GenerateMeshFromVolume(const vector<Voxel>& volume, int gridSize, float voxelSize, float isoValue = 0.0f) {
//	Mesh mesh;
//	int offsets[8][3] = {
//		{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
//		{0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}
//	};
//
//	// Marching Cubes lookup tables
//	extern int edgeTable[256];
//	extern int triTable[256][16];
//
//	// Iterate through the volume grid
//	for (int z = 0; z < gridSize - 1; ++z) {
//		for (int y = 0; y < gridSize - 1; ++y) {
//			for (int x = 0; x < gridSize - 1; ++x) {
//				// Gather voxel values for the cube
//				std::array<float, 8> values;
//				std::array<Eigen::Vector3f, 8> positions;
//				std::array<Eigen::Vector3f, 8> normals;
//				std::array<Eigen::Vector3f, 8> colors;
//
//				for (int i = 0; i < 8; ++i) {
//					int xi = x + offsets[i][0];
//					int yi = y + offsets[i][1];
//					int zi = z + offsets[i][2];
//					size_t index = zi * gridSize * gridSize + yi * gridSize + xi;
//
//					const auto& voxel = volume[index];
//					positions[i] = Eigen::Vector3f(xi, yi, zi) * voxelSize - Eigen::Vector3f(50.0f, 50.0f, 50.0f);
//					values[i] = voxel.tsdfValue;
//					normals[i] = voxel.normal;
//					colors[i] = voxel.color;
//				}
//
//				// Determine the cube index
//				int cubeIndex = 0;
//				for (int i = 0; i < 8; ++i) {
//					if (values[i] < isoValue) {
//						cubeIndex |= (1 << i);
//					}
//				}
//
//				// Skip empty cubes
//				if (edgeTable[cubeIndex] == 0) {
//					continue;
//				}
//
//				// Interpolate vertices
//				std::array<Eigen::Vector3f, 12> edgeVertices;
//				std::array<Eigen::Vector3f, 12> edgeNormals;
//				std::array<Eigen::Vector3f, 12> edgeColors;
//				for (int i = 0; i < 12; ++i) {
//					if (edgeTable[cubeIndex] & (1 << i)) {
//						int v1 = offsets[i][0];
//						int v2 = offsets[i][1];
//
//						float t = (isoValue - values[v1]) / (values[v2] - values[v1]);
//						edgeVertices[i] = positions[v1] + t * (positions[v2] - positions[v1]);
//						edgeNormals[i] = normals[v1] + t * (normals[v2] - normals[v1]);
//						edgeColors[i] = colors[v1] + t * (colors[v2] - colors[v1]);
//					}
//				}
//
//				// Add triangles to the mesh
//				for (int i = 0; triTable[cubeIndex][i] != -1; i += 3) {
//					int edge1 = triTable[cubeIndex][i];
//					int edge2 = triTable[cubeIndex][i + 1];
//					int edge3 = triTable[cubeIndex][i + 2];
//
//					unsigned int baseIndex = static_cast<unsigned int>(mesh.vertices.size());
//					mesh.vertices.push_back(edgeVertices[edge1]);
//					mesh.vertices.push_back(edgeVertices[edge2]);
//					mesh.vertices.push_back(edgeVertices[edge3]);
//
//					mesh.normals.push_back(edgeNormals[edge1].normalized());
//					mesh.normals.push_back(edgeNormals[edge2].normalized());
//					mesh.normals.push_back(edgeNormals[edge3].normalized());
//
//					mesh.colors.push_back(edgeColors[edge1].normalized());
//					mesh.colors.push_back(edgeColors[edge2].normalized());
//					mesh.colors.push_back(edgeColors[edge3].normalized());
//
//					mesh.indices.push_back(baseIndex);
//					mesh.indices.push_back(baseIndex + 1);
//					mesh.indices.push_back(baseIndex + 2);
//				}
//			}
//		}
//	}
//
//	return mesh;
//}

// Function to interpolate between two vertices
Eigen::Vector3f InterpolateVertex(const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, float v1, float v2, float isoValue) {
	if (std::abs(isoValue - v1) < 1e-6) return p1;
	if (std::abs(isoValue - v2) < 1e-6) return p2;
	if (std::abs(v1 - v2) < 1e-6) return p1;
	float t = (isoValue - v1) / (v2 - v1);
	return p1 + t * (p2 - p1);
}

// Generate mesh from volume using Marching Cubes
Mesh GenerateMeshFromVolume(const std::vector<Voxel>& volume, int gridSize, float voxelSize, float isoValue = 0.0f) {
	Mesh mesh;

	// Ensure volume size is correct
	if (volume.size() != static_cast<size_t>(gridSize * gridSize * gridSize)) {
		throw std::runtime_error("Volume size does not match grid dimensions.");
	}

	// Offsets for cube vertices
	int offsets[8][3] = {
		{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
		{0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}
	};

	// Iterate through the voxel grid
	for (int z = 0; z < gridSize - 1; ++z) {
		for (int y = 0; y < gridSize - 1; ++y) {
			for (int x = 0; x < gridSize - 1; ++x) {
				// Gather voxel values and attributes for the cube
				std::array<float, 8> values;
				std::array<Eigen::Vector3f, 8> positions;
				std::array<Eigen::Vector3f, 8> normals;
				std::array<Eigen::Vector3f, 8> colors;

				for (int i = 0; i < 8; ++i) {
					int xi = x + offsets[i][0];
					int yi = y + offsets[i][1];
					int zi = z + offsets[i][2];
					size_t index = zi * gridSize * gridSize + yi * gridSize + xi;

					const auto& voxel = volume[index];
					positions[i] = Eigen::Vector3f(xi, yi, zi) * voxelSize - Eigen::Vector3f(50.0f, 50.0f, 50.0f);
					values[i] = voxel.tsdfValue;
					normals[i] = voxel.normal;
					colors[i] = voxel.color;
				}

				// Determine cube index
				int cubeIndex = 0;
				for (int i = 0; i < 8; ++i) {
					if (values[i] < isoValue) {
						cubeIndex |= (1 << i);
					}
				}

				// Skip empty or full cubes
				if (edgeTable[cubeIndex] == 0) {
					continue;
				}

				// Interpolate vertices for active edges
				std::array<Eigen::Vector3f, 12> edgeVertices;
				std::array<Eigen::Vector3f, 12> edgeNormals;
				std::array<Eigen::Vector3f, 12> edgeColors;

				for (int i = 0; i < 12; ++i) {
					if (edgeTable[cubeIndex] & (1 << i)) {
						int v1 = i / 2;
						int v2 = i % 2;

						edgeVertices[i] = InterpolateVertex(positions[v1], positions[v2], values[v1], values[v2], isoValue);
						edgeNormals[i] = normals[v1] + normals[v2];
						edgeColors[i] = colors[v1] + colors[v2];
					}
				}

				// Generate triangles
				for (int i = 0; triTable[cubeIndex][i] != -1; i += 3) {
					int edge1 = triTable[cubeIndex][i];
					int edge2 = triTable[cubeIndex][i + 1];
					int edge3 = triTable[cubeIndex][i + 2];

					unsigned int baseIndex = static_cast<unsigned int>(mesh.vertices.size());

					mesh.vertices.push_back(edgeVertices[edge1]);
					mesh.vertices.push_back(edgeVertices[edge2]);
					mesh.vertices.push_back(edgeVertices[edge3]);

					mesh.normals.push_back(edgeNormals[edge1].normalized());
					mesh.normals.push_back(edgeNormals[edge2].normalized());
					mesh.normals.push_back(edgeNormals[edge3].normalized());

					mesh.colors.push_back(edgeColors[edge1].normalized());
					mesh.colors.push_back(edgeColors[edge2].normalized());
					mesh.colors.push_back(edgeColors[edge3].normalized());

					mesh.indices.push_back(baseIndex);
					mesh.indices.push_back(baseIndex + 1);
					mesh.indices.push_back(baseIndex + 2);
				}
			}
		}
	}

	return mesh;
}

void AppStartCallback_Simple(App* pApp)
{
	auto renderer = pApp->GetRenderer();
	//LoadModel(renderer, "C:\\Resources\\3D\\PLY\\Complete\\Lower.ply");

	//VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 100.0f, 0.0f, 0.0f }, Color4::Red);
	//VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 100.0f, 0.0f }, Color4::Green);
	//VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, 100.0f }, Color4::Blue);

	vtkNew<vtkPLYReader> reader;
	//reader->SetFileName("C:\\Resources\\3D\\PLY\\Complete\\Lower_pointcloud.ply");
	reader->SetFileName("./../../res/3D/Lower_pointcloud.ply");
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

	float voxelSize = 0.1f;
	float truncationDistance = 1.0f;
	float weight = 1.0f;
	vector<Voxel> volume;
	volume.resize(1000 * 1000 * 1000);

	t = Time::End(t, "Initialize volume");

	for (size_t pi = 0; pi < plyPoints->GetNumberOfPoints(); pi++)
	{
		pointIndices.push_back((unsigned int)pi);

		auto dp = plyPoints->GetPoint(pi);
		auto normal = plyNormals->GetTuple(pi);
		unsigned char color[3];
		plyColors->GetTypedTuple(pi, color);

		Eigen::Vector3f point((float)dp[0] - center.x(), (float)dp[1] - center.y(), (float)dp[2] - center.z());
		points.push_back(point);
		Eigen::Vector3f pointNormal((float)normal[0], (float)normal[1], (float)normal[2]);
		normals.push_back(pointNormal);
		auto color4 = Color4(color[0], color[1], color[2], 255);
		colors.push_back(color4);
		Eigen::Vector3f pointColor((float)color4.x() / 255.0f, (float)color4.y() / 255.0f, (float)color4.z() / 255.0f);

		//VD::AddSphere("points",
		//	point,
		//	{ 0.1f,0.1f,0.1f },
		//	{ 0.0f, 0.0f, 1.0f },
		//	Color4(color[0], color[1], color[2], 255));

		int xIndex = (int)floorf((point.x() + 50.0f) / voxelSize);
		int yIndex = (int)floorf((point.y() + 50.0f) / voxelSize);
		int zIndex = (int)floorf((point.z() + 50.0f) / voxelSize);

		for (int z = zIndex - 5; z <= zIndex + 5; z++)
		{
			for (int y = yIndex - 5; y <= yIndex + 5; y++)
			{
				for (int x = xIndex - 5; x <= xIndex + 5; x++)
				{
					size_t index = z * 1000 * 1000 + y * 1000 + x;

					Eigen::Vector3f position((float)x* voxelSize - 50.0f, (float)y* voxelSize - 50.0f, (float)z* voxelSize - 50.0f);
					float distance = (position - point).norm();

					float tsdfValue = 0.0f;
					if (distance <= truncationDistance) {
						tsdfValue = distance / truncationDistance;
						if ((position - point).dot(point) < 0.0f) {
							tsdfValue = -tsdfValue;
						}


						auto& voxel = volume[index];

						float oldTSDF = voxel.tsdfValue;
						if (FLT_MAX == oldTSDF)
						{
							voxel.tsdfValue = tsdfValue;
							voxel.weight = 1.0f;
							voxel.normal = pointNormal;
							voxel.color = pointColor;
						}
						else
						{
							float oldWeight = voxel.weight;
							float newTSDF = (oldTSDF * oldWeight + tsdfValue * weight) / (oldWeight + weight);
							float newWeight = oldWeight + weight;
							if (fabsf(newTSDF) < fabsf(oldTSDF))
							{
								voxel.tsdfValue = newTSDF;
								voxel.weight = oldWeight + weight;

								Eigen::Vector3f oldNormal = voxel.normal;
								Eigen::Vector3f oldColor = voxel.color;
								voxel.normal = (oldNormal + pointNormal) * 0.5f;
								voxel.color = (oldColor + pointColor) * 0.5f;
							}
						}
					}
				}
			}
		}
	}

	t = Time::End(t, "Integrate points");

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

	auto mesh = GenerateMeshFromVolume(volume, 1000, 0.1f);

	printf("mesh.vertices : %llu\n", mesh.vertices.size());
	printf("mesh.indices : %llu\n", mesh.indices.size());
	printf("mesh.normals : %llu\n", mesh.normals.size());
	printf("mesh.colors : %llu\n", mesh.colors.size());

	for (size_t i = 0; i < mesh.indices.size() / 3; i++)
	{
		auto& i0 = mesh.indices[i * 3];
		auto& i1 = mesh.indices[i * 3 + 1];
		auto& i2 = mesh.indices[i * 3 + 2];

		auto& v0 = mesh.vertices[i * 3];
		auto& v1 = mesh.vertices[i * 3 + 1];
		auto& v2 = mesh.vertices[i * 3 + 2];

		auto& n0 = mesh.normals[i * 3];
		auto& n1 = mesh.normals[i * 3 + 1];
		auto& n2 = mesh.normals[i * 3 + 2];

		auto& c0 = mesh.colors[i * 3];
		auto& c1 = mesh.colors[i * 3 + 1];
		auto& c2 = mesh.colors[i * 3 + 2];

		VD::AddTriangle("Mesh", v0, v1, v2, Color4::White);
	}

	//VD::AddCube("AABC", { 0.0f, 0.0f, 0.0f }, axisMax * 0.5f, Color4::White);
	//printf("axisMax : %f\n", axisMax);

	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { axisMax * 0.5f, 0.0f, 0.0f }, Color4::Red);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, axisMax * 0.5f, 0.0f }, Color4::Green);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, axisMax * 0.5f }, Color4::Blue);

	t = Time::End(t, "Visualize");
}

void AppStartCallback(App* pApp)
{
	//AppStartCallback_Integrate(pApp);
	//AppStartCallback_Convert(pApp);
	//AppStartCallback_LoadPNT(pApp);
	//AppStartCallback_KDTree(pApp);
	//AppStartCallback_Octree(pApp);
	//AppStartCallback_Poisson(pApp);
	AppStartCallback_Simple(pApp);
}