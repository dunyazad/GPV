#include <Common.h>
#include <App/App.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include <Algorithm/Octree.hpp>

#include <Algorithm/CustomPolyDataFilter.h>
#include <Algorithm/vtkMedianFilter.h>
#include <Algorithm/vtkQuantizingFilter.h>

#include <Debugging/VisualDebugging.h>

#include <CUDA/HashTable.cuh>
#include <CUDA/KDTree.cuh>
#include <CUDA/KDITree.cuh>
#include <CUDA/SVO.cuh>
#include <CUDA/AOctree.cuh>

#include <CUDA/Processing.cuh>

#include <CUDA/CUDA.cuh>

int depthIndex = 0;

int pid = 0;
size_t size_0 = 0;
size_t size_45 = 0;
float transform_0[16];
float transform_45[16];
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
	actor->SetObjectName("points");
	actor->SetMapper(mapper);

	actor->GetProperty()->SetPointSize(5.0f);
	actor->GetProperty()->SetColor(1.0, 1.0, 1.0);

	renderer->AddActor(actor);
}

void OnKeyPress(vtkObject* caller, long unsigned int eventId, void* clientData, void* callData) {
	vtkRenderWindowInteractor* interactor = static_cast<vtkRenderWindowInteractor*>(caller);
	vtkRenderWindow* renderWindow = interactor->GetRenderWindow();
	vtkRenderer* renderer = renderWindow->GetRenderers()->GetFirstRenderer();

	std::string key = interactor->GetKeySym();

	printf("%s\n", key.c_str());

	if (key == "r")
	{
		std::cout << "Key 'r' was pressed. Resetting camera." << std::endl;
		vtkRenderer* renderer = static_cast<vtkRenderer*>(clientData);

		vtkCamera* camera = renderer->GetActiveCamera();
		renderer->ResetCamera();

		// Reset the camera position, focal point, and view up vector to their defaults
		camera->SetPosition(0, 0, 1);          // Reset position to default
		camera->SetFocalPoint(0, 0, 0);        // Reset focal point to origin
		camera->SetViewUp(0, 1, 0);            // Reset the up vector to default (Y-axis up)

		interactor->Render();
	}
	else if (key == "c")
	{
		vtkRenderer* renderer = static_cast<vtkRenderer*>(clientData);

		vtkCamera* camera = renderer->GetActiveCamera();
		camera->SetParallelProjection(!camera->GetParallelProjection());

		interactor->Render();
	}
	else if (key == "Escape")
	{
		std::cout << "Key 'Escape' was pressed. Exiting." << std::endl;
		interactor->TerminateApp();
	}
	else if (key == "minus")
	{
		VisualDebugging::SetLineWidth("Spheres", 1);
		vtkSmartPointer<vtkActor> actor = VisualDebugging::GetSphereActor("Spheres");
		vtkSmartPointer<vtkMapper> mapper = actor->GetMapper();
		vtkSmartPointer<vtkPolyDataMapper> polyDataMapper =
			vtkPolyDataMapper::SafeDownCast(mapper);
		vtkSmartPointer<vtkGlyph3DMapper> glyph3DMapper = vtkGlyph3DMapper::SafeDownCast(mapper);
		vtkSmartPointer<vtkPolyData> polyData = vtkPolyData::SafeDownCast(glyph3DMapper->GetInputDataObject(0, 0));
		vtkSmartPointer<vtkPointData> pointData = polyData->GetPointData();
		vtkSmartPointer<vtkDoubleArray> scaleArray =
			vtkDoubleArray::SafeDownCast(pointData->GetArray("Scales"));
		for (vtkIdType i = 0; i < scaleArray->GetNumberOfTuples(); ++i)
		{
			double scale[3]; // Assuming 3-component scale array (X, Y, Z)
			scaleArray->GetTuple(i, scale);
			//std::cout << "Scale for point " << i << ": "
			//	<< scale[0 ] << ", " << scale[1] << ", " << scale[2] << std::endl;
			scale[0] *= 0.9;
			scale[1] *= 0.9;
			scale[2] *= 0.9;
			scaleArray->SetTuple(i, scale);
		}
		polyData->Modified();
		glyph3DMapper->SetScaleArray("Scales");
		glyph3DMapper->Update();
	}
	else if (key == "equal")
	{
		VisualDebugging::SetLineWidth("Spheres", 1);
		vtkSmartPointer<vtkActor> actor = VisualDebugging::GetSphereActor("Spheres");
		vtkSmartPointer<vtkMapper> mapper = actor->GetMapper();
		vtkSmartPointer<vtkPolyDataMapper> polyDataMapper =
			vtkPolyDataMapper::SafeDownCast(mapper);
		vtkSmartPointer<vtkGlyph3DMapper> glyph3DMapper = vtkGlyph3DMapper::SafeDownCast(mapper);
		vtkSmartPointer<vtkPolyData> polyData = vtkPolyData::SafeDownCast(glyph3DMapper->GetInputDataObject(0, 0));
		vtkSmartPointer<vtkPointData> pointData = polyData->GetPointData();
		vtkSmartPointer<vtkDoubleArray> scaleArray =
			vtkDoubleArray::SafeDownCast(pointData->GetArray("Scales"));
		for (vtkIdType i = 0; i < scaleArray->GetNumberOfTuples(); ++i)
		{
			double scale[3]; // Assuming 3-component scale array (X, Y, Z)
			scaleArray->GetTuple(i, scale);
			//std::cout << "Scale for point " << i << ": "
			//	<< scale[0 ] << ", " << scale[1] << ", " << scale[2] << std::endl;
			scale[0] *= 1.1;
			scale[1] *= 1.1;
			scale[2] *= 1.1;
			scaleArray->SetTuple(i, scale);
		}
		polyData->Modified();
		glyph3DMapper->SetScaleArray("Scales");
		glyph3DMapper->Update();
	}
	else if (key == "Return")
	{
		vtkRenderer* renderer = static_cast<vtkRenderer*>(clientData);
		vtkRenderWindow* renderWindow = renderer->GetRenderWindow();

		// Grab the depth buffer (z-values)
		int width = renderWindow->GetSize()[0];
		int height = renderWindow->GetSize()[1];
		vtkSmartPointer<vtkFloatArray> depthBuffer = vtkSmartPointer<vtkFloatArray>::New();
		depthBuffer->SetNumberOfComponents(1);
		depthBuffer->SetNumberOfTuples(width * height);

		renderWindow->GetZbufferData(0, 0, width - 1, height - 1, depthBuffer);

		// Save depth map to an image
		float minDepth = std::numeric_limits<float>::max();
		float maxDepth = std::numeric_limits<float>::lowest();

		for (vtkIdType i = 0; i < depthBuffer->GetNumberOfTuples(); i++) {
			float depthValue = depthBuffer->GetValue(i);
			minDepth = std::min(minDepth, depthValue);
			maxDepth = std::max(maxDepth, depthValue);
		}

		vtkSmartPointer<vtkImageData> depthImage = vtkSmartPointer<vtkImageData>::New();
		depthImage->SetDimensions(width, height, 1);
		depthImage->AllocateScalars(VTK_FLOAT, 1);

		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				float depthValue = depthBuffer->GetValue((height - y - 1) * width + x);
				float normalizedDepth = (depthValue - minDepth) / (maxDepth - minDepth);

				depthImage->SetScalarComponentFromFloat(x, y, 0, 0, normalizedDepth * 255);

				if (0.0f != normalizedDepth)
				{
					VisualDebugging::AddSphere("Depth", { (float)x / (float)width * 100.0f, (float)y / (float)height * 100.0f, normalizedDepth * 100.0f }, { 0.1f, 0.1f, 0.1f }, { 0.0f, 0.0f, 0.0f }, Color4::White);
				}
			}
		}

		// Optional: Cast the float image to unsigned char to save as PNG
		vtkSmartPointer<vtkImageCast> castFilter = vtkSmartPointer<vtkImageCast>::New();
		castFilter->SetInputData(depthImage);
		castFilter->SetOutputScalarTypeToUnsignedChar();
		castFilter->Update();

		vtkSmartPointer<vtkPNGWriter> writer = vtkSmartPointer<vtkPNGWriter>::New();
		writer->SetFileName("c:\\Debug\\2D\\depthmap.png");
		writer->SetInputData(castFilter->GetOutput());
		writer->Write();
	}
	else if (key == "1")
	{
		VisualDebugging::ToggleVisibility("Original");
	}
	else if (key == "2")
	{
		VisualDebugging::ToggleVisibility("DownSample");
	}
	else if (key == "3")
	{
		VisualDebugging::ToggleVisibility("Filter");
	}
	else if (key == "4")
	{
		VisualDebugging::ToggleVisibility("OctreeNode");
	}
	else if (key == "5")
	{
		VisualDebugging::ToggleVisibility("WiredBox_4");
	}
	else if (key == "6")
	{
		VisualDebugging::ToggleVisibility("WiredBox_5");
	}
	else if (key == "7")
	{
		VisualDebugging::ToggleVisibility("WiredBox_6");
	}
	else if (key == "8")
	{
		VisualDebugging::ToggleVisibility("WiredBox_7");
	}
	else if (key == "9")
	{
		VisualDebugging::ToggleVisibility("WiredBox_8");
	}
	else if (key == "0")
	{
		VisualDebugging::ToggleVisibility("WiredBox_9");
	}
	else if (key == "Left")
	{
		depthIndex--;
		if (depthIndex < 0) depthIndex = 0;

		for (int i = 0; i < 14; i++)
		{
			stringstream ss;
			ss << "Cubes_" << i;
			VisualDebugging::SetVisibility(ss.str(), false);
		}
		{
			stringstream ss;
			ss << "Cubes_" << depthIndex;
			VisualDebugging::SetVisibility(ss.str(), true);
		}

		printf("Depth : %d\n", depthIndex);
	}
	else if (key == "Right")
	{
		depthIndex++;
		if (depthIndex > 14) depthIndex = 13;

		for (int i = 0; i < 14; i++)
		{
			stringstream ss;
			ss << "Cubes_" << i;
			VisualDebugging::SetVisibility(ss.str(), false);
		}
		{
			stringstream ss;
			ss << "Cubes_" << depthIndex;
			VisualDebugging::SetVisibility(ss.str(), true);
		}

		printf("Depth : %d\n", depthIndex);
	}
	else if (key == "space")
	{
	}
}

int main()
{
	App app;
	app.AddAppStartCallback([&](App* pApp) {
		auto renderer = pApp->GetRenderer();

		{
			Processing::PatchProcessor pp;
			float sampleSize = 0.1f;
			pp.Initialize(2000, 2000, 256 * 480);

			int i = 3;
			//for (int i = 3; i < 244; i++)
				//for (int i = 3; i < 57; i++)
				//for (int cnt = 0; cnt < 4; cnt++)
			{
				printf("======= [%4d] =======\n", i);

				auto te = Time::Now();
				LoadPatch(i, renderer);
				Time::End(te, "Loading PointCloud Patch");
			}

			//{
			//	float sampleSize = 0.1f;

			//	printf("[[[ patchPoints.size() ]]] : %llu\n", patchPoints_0.size());

			//	auto t = Time::Now();

			//	pp.DownSample(patchPoints_0.data(), patchPoints_0.size(), sampleSize, aabb_0.min(), aabb_0.max());

			//	t = Time::End(t, "DownSampling");

			//	printf("[[[ *pp.h_numberOfResultPoints ]]] : %llu\n", pp.h_numberOfResultPoints);

			//	//for (size_t i = 0; i < pp.h_numberOfResultPoints; i++)
			//	//{
			//	//	auto& p = pp.h_resultPoints[i];

			//	//	//p += Eigen::Vector3f(0.0f, 0.0f, 10.0f);

			//	//	VisualDebugging::AddSphere("DownSample", p, { 0.05f, 0.05f, 0.05f }, { 0.0f, 0.0f, 0.0f }, Color4::Blue);
			//	//}
			//}

			//{
			//	float sampleSize = 0.1f;

			//	printf("[[[ patchPoints.size() ]]] : %llu\n", patchPoints_0.size());

			//	auto t = Time::Now();

			//	pp.MedianFilter(patchPoints_0.data(), patchPoints_0.size(), sampleSize, aabb_45.min(), aabb_45.max());

			//	t = Time::End(t, "MedianFilter");

			//	printf("[[[ *pp.h_numberOfResultPoints ]]] : %llu\n", pp.h_numberOfResultPoints);

			//	//for (size_t i = 0; i < pp.h_numberOfResultPoints; i++)
			//	//{
			//	//	auto& p = pp.h_resultPoints[i];

			//	//	//p += Eigen::Vector3f(0.0f, 0.0f, 10.0f);

			//	//	VisualDebugging::AddSphere("Filter", p, { 0.05f, 0.05f, 0.05f }, { 0.0f, 0.0f, 0.0f }, Color4::Red);
			//	//}
			//}
		}

		vtkNew<vtkPoints> points;
		for (size_t i = 0; i < inputPoints.size(); i++)
		{
			auto& p = inputPoints[i];

			VisualDebugging::AddSphere("Original", p, { 0.05f, 0.05f, 0.05f }, { 0.0f, 0.0f, 0.0f }, Color4::Red);

			points->InsertNextPoint(p.x(), p.y(), p.z());
		}

		vtkNew<vtkPolyData> polyData;
		polyData->SetPoints(points);

		vtkNew<vtkKdTreePointLocator> octree;
		octree->SetDataSet(polyData);
		octree->BuildLocator();

		vtkNew<vtkImageData> volume;
		int dimensions[3] = { 50, 50, 50 };  // Adjust grid resolution as necessary
		volume->SetDimensions(dimensions);
		//volume->SetOrigin(-2, -2, -2);     // Adjust origin based on your point cloud
		volume->SetOrigin(0, 0, 0);
		volume->SetSpacing(0.1, 0.1, 0.1); // Adjust spacing

		vtkNew<vtkFloatArray> scalars;
		scalars->SetNumberOfComponents(1);
		scalars->SetNumberOfTuples(dimensions[0] * dimensions[1] * dimensions[2]);
		volume->GetPointData()->SetScalars(scalars);

		// Populate volume data using the octree to set occupancy
		for (int z = 0; z < dimensions[2]; ++z)
		{
			for (int y = 0; y < dimensions[1]; ++y)
			{
				for (int x = 0; x < dimensions[0]; ++x)
				{
					double p[3];
					volume->GetPoint(x + y * dimensions[0] + z * dimensions[0] * dimensions[1], p);

					vtkIdType ptId = octree->FindClosestPoint(p);
					double closestPt[3];
					octree->GetDataSet()->GetPoint(ptId, closestPt);

					double distanceSquared = vtkMath::Distance2BetweenPoints(p, closestPt);
					double threshold = 0.01; // Adjust threshold as needed
					scalars->SetValue(x + y * dimensions[0] + z * dimensions[0] * dimensions[1],
						distanceSquared < threshold ? 1.0 : 0.0);
				}
			}
		}

		vtkNew<vtkMarchingCubes> marchingCubes;
		marchingCubes->SetInputData(volume);
		marchingCubes->SetValue(0, 1.0); // Iso-surface value; adjust based on voxel values
		marchingCubes->Update();

		vtkNew<vtkPolyDataMapper> mapper;
		mapper->SetInputConnection(marchingCubes->GetOutputPort());

		vtkNew<vtkActor> actor;
		actor->SetMapper(mapper);

		renderer->AddActor(actor);

		VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 100.0f, 0.0f, 0.0f }, Color4::Red);
		VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 100.0f, 0.0f }, Color4::Green);
		VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, 100.0f }, Color4::Blue);
		});

	app.AddKeyPressCallback(OnKeyPress);
	app.Run();

	return 0;
}
