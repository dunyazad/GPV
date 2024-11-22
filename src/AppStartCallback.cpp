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

void AppStartCallback(App* pApp)
{
	//AppStartCallback_Integrate(pApp);
	//AppStartCallback_Convert(pApp);
	//AppStartCallback_LoadPNT(pApp);
	//AppStartCallback_KDTree(pApp);
	AppStartCallback_Octree(pApp);
}