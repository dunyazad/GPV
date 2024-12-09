#include <App/AppStartCallback.h>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

//
//struct Point
//{
//	Eigen::Vector3f position;
//	Eigen::Vector3f normal;
//	Eigen::Vector3f color;
//};
//
//void AppStartCallback_Convert(App* pApp)
//{
//	auto renderer = pApp->GetRenderer();
//	Point* points = new Point[256 * 480];
//	size_t numberOfPoints = 0;
//
//	for (size_t i = 0; i < 4252; i++)
//	{
//		auto te = Time::Now();
//
//		numberOfPoints = 0;
//		memset(points, 0, sizeof(Point) * 256 * 480);
//
//		stringstream ss;
//		ss << "C:\\Resources\\2D\\Captured\\PointCloud\\point_" << i << ".ply";
//
//		vtkNew<vtkPLYReader> reader;
//		reader->SetFileName(ss.str().c_str());
//		reader->Update();
//
//		vtkPolyData* polyData = reader->GetOutput();
//
//		auto plyPoints = polyData->GetPoints();
//		vtkDataArray* plyNormals = polyData->GetPointData()->GetNormals();
//		vtkUnsignedCharArray* plyColors = vtkUnsignedCharArray::SafeDownCast(polyData->GetPointData()->GetScalars());
//
//		for (size_t pi = 0; pi < plyPoints->GetNumberOfPoints(); pi++)
//		{
//			auto dp = plyPoints->GetPoint(pi);
//			auto normal = plyNormals->GetTuple(pi);
//			unsigned char color[3];
//			plyColors->GetTypedTuple(pi, color);
//
//			points[pi].position.x() = dp[0];
//			points[pi].position.y() = dp[1];
//			points[pi].position.z() = dp[2];
//
//			points[pi].normal.x() = normal[0];
//			points[pi].normal.y() = normal[1];
//			points[pi].normal.z() = normal[2];
//
//			points[pi].color.x() = (float)color[0] / 255.0f;
//			points[pi].color.y() = (float)color[1] / 255.0f;
//			points[pi].color.z() = (float)color[2] / 255.0f;
//
//			numberOfPoints++;
//		}
//
//		stringstream oss;
//		oss << "C:\\Debug\\Patches\\point_" << i << ".pnt";
//		ofstream ofs;
//		ofs.open(oss.str(), ios::out | ios::binary);
//
//		ofs.write((char*)&numberOfPoints, sizeof(size_t));
//		ofs.write((char*)points, numberOfPoints * sizeof(Point));
//		ofs.close();
//
//		Time::End(te, "Convering PointCloud Patch", i);
//	}
//
//	delete[] points;
//}
//
//void AppStartCallback_LoadPNT(App* pApp)
//{
//	auto renderer = pApp->GetRenderer();
//	Point* points = new Point[256 * 480];
//	size_t numberOfPoints = 0;
//
//	float voxelSize = 0.1f;
//	int volumeDimensionX = 1000;
//	int volumeDimensionY = 1000;
//	int volumeDimensionZ = 500;
//
//	LoadModel(renderer, "C:\\Resources\\3D\\PLY\\Complete\\Lower.ply");
//
//	//SaveTRNFile();
//	LoadTRNFile();
//
//	Eigen::Vector3f modelTranslation(20.0f, 75.0f, 25.0f);
//
//	//CUDA::Voxel* volume;
//	//cudaMallocManaged(&volume, sizeof(CUDA::Voxel) * volumeDimensionX * volumeDimensionY * volumeDimensionZ);
//	//cudaDeviceSynchronize();
//
//	//Eigen::Vector3f* inputPoints = nullptr;
//	//cudaMallocManaged(&inputPoints, sizeof(Eigen::Vector3f) * 256 * 480);
//
//	//for (size_t i = 0; i < 4252; i++)
//	//{
//	//	auto te = Time::Now();
//
//	//	memset(points, 0, sizeof(Point) * 256 * 480);
//
//	//	stringstream ss;
//	//	ss << "C:\\Debug\\Patches\\point_" << i << ".pnt";
//	//	ifstream ifs;
//	//	ifs.open(ss.str(), ios::in | ios::binary);
//	//	ifs.read((char*)&numberOfPoints, sizeof(size_t));
//	//	ifs.read((char*)points, numberOfPoints * sizeof(Point));
//	//	ifs.close();
//
//	//	for (size_t pi = 0; pi < numberOfPoints; pi++)
//	//	{
//	//		auto& p = points[pi].position;
//	//		auto& n = points[pi].normal;
//	//		auto& c = points[pi].color;
//	//		p += modelTranslation;
//
//	//		inputPoints[pi] = p;
//	//	}
//
//	//	CUDA::IntegrateInputPoints(
//	//		volume,
//	//		make_int3(volumeDimensionX, volumeDimensionY, volumeDimensionZ),
//	//		0.1f,
//	//		inputPoints,
//	//		numberOfPoints);
//
//	//	Time::End(te, "Loading PointCloud Patch", i);
//	//}
//	//delete[] points;
//
//	//auto t = Time::Now();
//	//for (size_t i = 0; i < volumeDimensionX * volumeDimensionY * volumeDimensionZ; i++)
//	//{
//	//	if (volume[i].tsdfValue != 1.0f) continue;
//
//	//	int zKey = i / (volumeDimensionX * volumeDimensionY);
//	//	int yKey = (i % (volumeDimensionX * volumeDimensionY)) / volumeDimensionX;
//	//	int xKey = (i % (volumeDimensionX * volumeDimensionY)) % volumeDimensionX;
//
//	//	float x = xKey * voxelSize - modelTranslation.x();
//	//	float y = yKey * voxelSize - modelTranslation.y();
//	//	float z = zKey * voxelSize - modelTranslation.z();
//
//	//	//printf("%f, %f, %f\n", x, y, z);
//
//	//	VD::AddCube("temp", { x, y, z }, 0.1f, Color4::Red);
//	//}
//	//Time::End(t, "Show Voxels");
//
//	//cudaFree(volume);
//
//	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 100.0f, 0.0f, 0.0f }, Color4::Red);
//	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 100.0f, 0.0f }, Color4::Green);
//	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, 100.0f }, Color4::Blue);
//}
//
void AppStartCallback_Integrate(App* pApp)
{
	auto renderer = pApp->GetRenderer();

	float voxelSize = 0.1f;
	int volumeDimensionX = 1000;
	int volumeDimensionY = 1000;
	int volumeDimensionZ = 500;

	//LoadModel(renderer, "C:\\Resources\\3D\\PLY\\Complete\\Lower.ply");

	//SaveTRNFile();
	LoadTRNFile();

	Eigen::Vector3f modelTranslation(20.0f, 75.0f, 25.0f);

	uint3 volumeDimension = make_uint3(volumeDimensionX, volumeDimensionY, volumeDimensionZ);

	CUDA::cuCache::Voxel* volume;
	cudaMallocManaged(&volume, sizeof(CUDA::RegularGrid::Voxel) * volumeDimensionX * volumeDimensionY * volumeDimensionZ);
	cudaDeviceSynchronize();

	Eigen::Vector3f* inputPoints = nullptr;
	cudaMallocManaged(&inputPoints, sizeof(Eigen::Vector3f) * 256 * 480);

	Eigen::Vector3f* inputNormals = nullptr;
	cudaMallocManaged(&inputNormals, sizeof(Eigen::Vector3f) * 256 * 480);

	Eigen::Vector3f* inputColors = nullptr;
	cudaMallocManaged(&inputColors, sizeof(Eigen::Vector3f) * 256 * 480);

	size_t numberOfInputPoints = 0;

	/*
	{
		float voxelSize = 0.1f;
		int volumeDimensionX = 1000;
		int volumeDimensionY = 1000;
		int volumeDimensionZ = 1000;

		auto t = Time::Now();

		Eigen::AlignedBox3f aabb;
		vector<Eigen::Vector3f> loadedPoints;
		int loadedCount = 0;

		//size_t i = 3;
		//for (size_t i = 400; i < 500; i++)
		for (size_t i = 0; i < 4252; i++)
		{
			auto te = Time::Now();

			stringstream ss;
			ss << "C:\\Resources\\2D\\Captured\\PointCloud\\point_" << i << ".ply";

			vtkNew<vtkPLYReader> reader;
			reader->SetFileName(ss.str().c_str());
			reader->Update();

			vtkPolyData* polyData = reader->GetOutput();

			auto points = polyData->GetPoints();
			for (size_t pi = 0; pi < points->GetNumberOfPoints(); pi++)
			{
				auto dp = points->GetPoint(pi);
				//Eigen::Vector4f tp = cameraTransforms[i] * Eigen::Vector4f(dp[0], dp[1], dp[2] + 20.0f, 1.0f);
				//Eigen::Vector3f p(tp.x(), tp.y(), tp.z());

				Eigen::Vector3f p(dp[0], dp[1], dp[2]);
				p += modelTranslation;

				int xKey = (int)floorf(p.x() / voxelSize);
				int yKey = (int)floorf(p.y() / voxelSize);
				int zKey = (int)floorf(p.z() / voxelSize);

				if (xKey < 0 || xKey >= volumeDimensionX) continue;
				if (yKey < 0 || yKey >= volumeDimensionY) continue;
				if (zKey < 0 || zKey >= volumeDimensionZ) continue;

				volume[zKey * volumeDimensionX * volumeDimensionY + yKey * volumeDimensionX + xKey].tsdfValue = 1.0f;
				//VD::AddCube("temp", p, 0.1f, Color4::Red);
			}

			Time::End(te, "Loading PointCloud Patch", i);
		}

		t = Time::Now();
		for (size_t i = 0; i < volumeDimensionX * volumeDimensionY * volumeDimensionZ; i++)
		{
			if (volume[i].tsdfValue != 1.0f) continue;

			int zKey = i / (volumeDimensionX * volumeDimensionY);
			int yKey = (i % (volumeDimensionX * volumeDimensionY)) / volumeDimensionX;
			int xKey = (i % (volumeDimensionX * volumeDimensionY)) % volumeDimensionX;

			float x = xKey * voxelSize - modelTranslation.x();
			float y = yKey * voxelSize - modelTranslation.y();
			float z = zKey * voxelSize - modelTranslation.z();

			//printf("%f, %f, %f\n", x, y, z);

			VD::AddCube("temp", {x, y, z}, 0.1f, Color4::Red);
		}
		Time::End(t, "Show Voxels");

		//for (auto& kvp : volume)
		//{
		//	auto p = GetPosition(kvp.first);
		//	VD::AddCube("voxels", p, { 0.1f, 0.1f, 0.1f }, { 0.0f, 0.0f, 1.0f }, Color4::Red);
		//}
	}
	*/

	{
		auto t = Time::Now();

		Eigen::AlignedBox3f aabb;
		vector<Eigen::Vector3f> loadedPoints;
		int loadedCount = 0;

		CUDA::cuCache::ClearVolume(volume, volumeDimension);

		//size_t i = 3;
		for (size_t i = 0; i < 10; i++)
		//for (size_t i = 0; i < 4252; i++)
		{
			cudaMemset(inputPoints, 0.0f, sizeof(Eigen::Vector3f) * 256 * 480);
			numberOfInputPoints = 0;

			auto te = Time::Now();

			stringstream ss;
			ss << "C:\\Resources\\2D\\Captured\\PointCloud\\point_" << i << ".ply";

			vtkNew<vtkPLYReader> reader;
			reader->SetFileName(ss.str().c_str());
			reader->Update();

			vtkPolyData* polyData = reader->GetOutput();


			auto plyPoints = polyData->GetPoints();
			float* rawPoints = static_cast<float*>(plyPoints->GetData()->GetVoidPointer(0));
			vtkDataArray* plyNormals = polyData->GetPointData()->GetNormals();
			float* rawNormals = static_cast<float*>(plyNormals->GetVoidPointer(0));
			vtkUnsignedCharArray* plyColors = vtkUnsignedCharArray::SafeDownCast(polyData->GetPointData()->GetScalars());

			for (size_t pi = 0; pi < plyPoints->GetNumberOfPoints(); pi++)
			{
				auto dp = plyPoints->GetPoint(pi);
				auto normal = plyNormals->GetTuple(pi);
				unsigned char color[3];
				plyColors->GetTypedTuple(pi, color);
				//Eigen::Vector4f tp = cameraTransforms[i] * Eigen::Vector4f(dp[0], dp[1], dp[2] + 20.0f, 1.0f);
				//Eigen::Vector3f p(tp.x(), tp.y(), tp.z());

				Eigen::Vector3f p(dp[0], dp[1], dp[2]);
				p += modelTranslation;
				inputPoints[pi] = p;

				aabb.extend(p);

				Eigen::Vector3f n(normal[0], normal[1], normal[2]);
				inputNormals[pi] = n;

				Eigen::Vector3f c((float)color[0] / 255.0f, (float)color[1] / 255.0f, (float)color[2] / 255.0f);
				inputColors[pi] = c;

				numberOfInputPoints++;
			}

			CUDA::cuCache::IntegrateInputPoints(
				volume,
				volumeDimension,
				0.1f,
				numberOfInputPoints,
				inputPoints,
				inputNormals,
				inputColors);

			Time::End(te, "Loading PointCloud Patch", i);
		}

		auto numberOfSurfaceVoxels = CUDA::cuCache::GetNumberOfSurfaceVoxels(volume, volumeDimension, 0.1f);

		CUDA::cuCache::Point* resultPoints;
		cudaMallocManaged(&resultPoints, sizeof(CUDA::cuCache::Point) * numberOfSurfaceVoxels * 5);

		size_t* numberOfResultPoints;
		cudaMallocManaged(&numberOfResultPoints, sizeof(size_t));
		cudaMemset(numberOfResultPoints, 0, sizeof(size_t));

		CUDA::cuCache::ExtractSurfacePoints(volume, volumeDimension, voxelSize, aabb.min(), resultPoints, numberOfResultPoints);

		cudaFree(volume);

		cudaDeviceSynchronize();

		printf("Number of surface voxeles : %llu\n", numberOfSurfaceVoxels);
		printf("Number of result points : %llu\n", *numberOfResultPoints);

		PLYFormat ply;

		auto nop = *numberOfResultPoints;
		CUDA::cuCache::Point* h_resultPoints = new CUDA::cuCache::Point[nop];
		cudaMemcpy(h_resultPoints, resultPoints, sizeof(CUDA::cuCache::Point) * nop, cudaMemcpyDeviceToHost);

		for (size_t i = 0; i < nop; i++)
		{
			auto& p = h_resultPoints[i];

			Eigen::Vector3f tp = p.position - modelTranslation;

			ply.AddPointFloat3(tp.data());
			ply.AddNormalFloat3(p.normal.data());
			ply.AddColorFloat3(p.color.data());

			Color4 c;
			c.FromNormalized(p.color.x(), p.color.y(), p.color.z(), 1.0f);

			VD::AddCube("ResultPoints", p.position, { 0.1f, 0.1f, 0.1f }, {0.0f, 0.0f, 1.0f}, c);
		}

		ply.Serialize("C:\\Resources\\Debug\\Temp.ply");


		cudaFree(resultPoints);
		cudaFree(numberOfResultPoints);

		cudaDeviceSynchronize();

		t = Time::Now();
		for (size_t i = 0; i < volumeDimensionX * volumeDimensionY * volumeDimensionZ; i++)
		{
			if (-0.05f < volume[i].tsdfValue && volume[i].tsdfValue < 0.05f)
			{
				int zKey = i / (volumeDimensionX * volumeDimensionY);
				int yKey = (i % (volumeDimensionX * volumeDimensionY)) / volumeDimensionX;
				int xKey = (i % (volumeDimensionX * volumeDimensionY)) % volumeDimensionX;

				float x = xKey * voxelSize - modelTranslation.x();
				float y = yKey * voxelSize - modelTranslation.y();
				float z = zKey * voxelSize - modelTranslation.z();

				//printf("%f, %f, %f\n", x, y, z);

				Color4 color(255, 255, 255, 255);
				color.FromNormalized(volume[i].color.x(), volume[i].color.y(), volume[i].color.z(), 1.0f);

				VD::AddCube("temp", { x, y, z }, 0.1f, color);
			}
		}
		Time::End(t, "Show Voxels");

		//for (auto& kvp : volume)
		//{
		//	auto p = GetPosition(kvp.first);
		//	VD::AddCube("voxels", p, { 0.1f, 0.1f, 0.1f }, { 0.0f, 0.0f, 1.0f }, Color4::Red);
		//}
	}

	cudaFree(volume);

	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 100.0f, 0.0f, 0.0f }, Color4::Red);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 100.0f, 0.0f }, Color4::Green);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, 100.0f }, Color4::Blue);
}
