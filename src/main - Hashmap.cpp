#include <Common.h>
#include <App/App.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include <Algorithm/vtkMedianFilter.h>
#include <Algorithm/vtkQuantizingFilter.h>

#include <Debugging/VisualDebugging.h>

#include <CUDA/HashTable.cuh>

void OnKeyPress(vtkObject* caller, long unsigned int eventId, void* clientData, void* callData) {
	vtkRenderWindowInteractor* interactor = static_cast<vtkRenderWindowInteractor*>(caller);
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
	else if (key == "space")
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
					VisualDebugging::AddSphere("Depth", { (float)x / (float)width * 100.0f, (float)y / (float)height * 100.0f, normalizedDepth * 100.0f }, { 0.1f, 0.1f, 0.1f }, { 0.0f, 0.0f, 0.0f }, 255, 255, 255);
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
		VisualDebugging::ToggleVisibility("Spheres");
	}
	else if (key == "Right")
	{
	}
}

struct VolumeInfo
{
	Eigen::Vector3f volumeMin = Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
	Eigen::Vector3f volumeMax = Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
	float voxelSize = 0.1f;
};

struct SpatialIndex
{
	uint64_t x = UINT64_MAX;
	uint64_t y = UINT64_MAX;
	uint64_t z = UINT64_MAX;
};

struct HashKey
{
	uint64_t key = UINT64_MAX;
};

HashKey GetHashKey(const SpatialIndex& index)
{
	return
	{
		(((uint64_t)(index.x) & 0x000000000000FFFF) << 32) +
		(((uint64_t)(index.y) & 0x000000000000FFFF) << 16) +
		(((uint64_t)(index.z) & 0x000000000000FFFF) << 0)
	};
}

SpatialIndex GetSpatialIndex(const HashKey& hashKey)
{
	SpatialIndex index;
	index.x = (hashKey.key >> 32) & 0xffff;
	index.y = (hashKey.key >> 16) & 0xffff;
	index.z = (hashKey.key) & 0xffff;
	return index;
}

SpatialIndex GetSpatialIndex(const Eigen::Vector3f& position, const VolumeInfo& volumeInfo)
{
	auto local = position - volumeInfo.volumeMin;
	SpatialIndex index;
	index.x = (uint64_t)floorf(local.x() / volumeInfo.voxelSize);
	index.y = (uint64_t)floorf(local.y() / volumeInfo.voxelSize);
	index.z = (uint64_t)floorf(local.z() / volumeInfo.voxelSize);
	return index;
}

Eigen::Vector3f GetPosition(const SpatialIndex& index, const VolumeInfo& volumeInfo)
{
	return
	{
		volumeInfo.volumeMin.x() + ((float)index.x) * volumeInfo.voxelSize,
		volumeInfo.volumeMin.x() + ((float)index.x) * volumeInfo.voxelSize,
		volumeInfo.volumeMin.x() + ((float)index.x) * volumeInfo.voxelSize
	};
}

HashKey GetHashKey(const Eigen::Vector3f& position, const VolumeInfo& volumeInfo)
{
	return GetHashKey(GetSpatialIndex(position, volumeInfo));
}

Eigen::Vector3f GetPosition(const HashKey& hashKey, const VolumeInfo& volumeInfo)
{
	return GetPosition(GetSpatialIndex(hashKey), volumeInfo);
}

int main()
{
	printf("key : %llu\n", GetHashKey({ 2345, 4123, 1424 }).key);

	App app;
	app.AddAppStartCallback([](App* pApp) {
		VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 255.0f, 0, 0 }, 255, 0, 0);
		VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0, 255.0f, 0 }, 0, 255, 0);
		VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0, 0, 255.0f }, 0, 0, 255);

		{
			VolumeInfo volumeInfo
			{
				{-500.0f, -500.0f, -500.0f},
				{500.0f, 500.0f, 500.0f},
				0.1f
			};

			Algorithm::HashTable::KeyValue* pHashTable = Algorithm::HashTable::create_hashtable();
			vector<Algorithm::HashTable::KeyValue> kvps;

			vtkNew<vtkPLYReader> reader;
			reader->SetFileName("C:\\Resources\\Debug\\VoxelValues.ply");
			reader->Update();

			{
				vtkSmartPointer<vtkPolyData> polyData = reader->GetOutput();

				vtkPoints* points = polyData->GetPoints();
				vtkUnsignedCharArray* colors = vtkUnsignedCharArray::SafeDownCast(polyData->GetPointData()->GetScalars());

				if (!points)
				{
					std::cerr << "No points found in the PLY file." << std::endl;
					return;
				}

				for (vtkIdType i = 0; i < points->GetNumberOfPoints(); ++i)
				{
					double p[3];
					points->GetPoint(i, p);
					float x = (float)p[0];
					float y = (float)p[1];
					float z = (float)p[2];

					x = (float)((int)roundf(x * 100.0f + 5.0f)) / 100.0f;
					y = (float)((int)roundf(y * 100.0f + 5.0f)) / 100.0f;
					z = (float)((int)roundf(z * 100.0f + 5.0f)) / 100.0f;

					//std::cout << "Point " << i << ": (" << x << ", " << y << ", " << z << ")";

					float r = 1.0f;
					float g = 1.0f;
					float b = 1.0f;
					float a = 1.0f;

					float v = FLT_MAX;

					if (colors && colors->GetNumberOfComponents() == 4)
					{
						unsigned char color[4];
						colors->GetTypedTuple(i, color);
						//std::cout << ", Color: (" << static_cast<int>(color[0]) << ", "
						//	<< static_cast<int>(color[1]) << ", "
						//	<< static_cast<int>(color[2]) << ", "
						//	<< static_cast<int>(color[3]) << ")";

						r = (float)color[0] / 255.0f;
						g = (float)color[1] / 255.0f;
						b = (float)color[2] / 255.0f;
						a = (float)color[3] / 255.0f;

						//if (v > 1.0f) v = 1.0f;
						//if (v < -1.0f) v = -1.0f;
						//
						//r = v > 0.01f ? v : 0.0f;
						//b = fabsf(v) <= 0.01f ? 1.0f : 0.0f;
						//g = v < -0.01f ? -v : 0.0f;
						//a = 1.0f;

						if (r > 0.0f && b > 0.0f) v = 0.0f;
						else if (r > 0.0f) v = r;
						else if (b > 0.0f) v = g;
					}
					//std::cout << std::endl;
					//(unsigned char)((1.0f - fabsf(v)) * 255.0f)
					//VisualDebugging::AddSphere("Spheres", { x, y, z }, { 0.05f, 0.05f, 0.05f }, { 0, 1, 0 }, r, g, b);

					auto key = GetHashKey({ x, y, z }, volumeInfo);
					kvps.push_back({ key.key, (uint64_t)i });
				}
			}

			const uint64_t num_insert_batches = 16;
			uint64_t num_inserts_per_batch = (uint64_t)kvps.size() / num_insert_batches;
			for (uint64_t i = 0; i < num_insert_batches; i++)
			{
				insert_hashtable(pHashTable, kvps.data() + i * num_inserts_per_batch, num_inserts_per_batch);

				printf("i : %llu\n", i);
			}


			vtkNew<vtkVertexGlyphFilter> glyphFilter;
			glyphFilter->SetInputConnection(reader->GetOutputPort());
			glyphFilter->Update();

			vtkNew<vtkPolyDataMapper> mapper;
			mapper->SetInputConnection(glyphFilter->GetOutputPort());

			vtkNew<vtkActor> actor;
			actor->SetMapper(mapper);

			//actor->GetProperty()->BackfaceCullingOff();  // Ensure back faces are visible
			//actor->GetProperty()->SetLighting(true);     // Ensure lighting is on
			actor->GetProperty()->SetLighting(false);     // Ensure lighting is on

			//pApp->GetRenderer()->AddActor(actor);
		}
		});
	app.AddKeyPressCallback(OnKeyPress);
	app.Run();

	return 0;
}
