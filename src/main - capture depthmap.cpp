#include <Common.h>

#include <openvdb/openvdb.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include <App/CustomTrackballStyle.h>
#include <App/Utility.h>

#include <Algorithm/vtkMedianFilter.h>
#include <Algorithm/vtkQuantizingFilter.h>

#include <Debugging/VisualDebugging.h>

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
}

class TimerCallback : public vtkCommand
{
public:
    static TimerCallback* New() { return new TimerCallback; }

    TimerCallback() = default;

    virtual void Execute(vtkObject* caller, unsigned long eventId, void* vtkNotUsed(callData)) override
    {
        if (eventId == vtkCommand::TimerEvent) {
            animate();
        }
        else {
            std::cerr << "Unexpected event ID: " << eventId << std::endl;
        }
    }

private:
    void animate() { VisualDebugging::Update(); }
};

int main() {
    openvdb::initialize();

    MaximizeConsoleWindowOnMonitor(1);

    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->SetBackground(0.3, 0.5, 0.7);

    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    //renderWindow->SetSize(1920, 1080);
	renderWindow->SetSize(256, 480);
    renderWindow->AddRenderer(renderer);

    vtkSmartPointer<vtkRenderWindowInteractor> interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    //vtkNew<vtkInteractorStyleTrackballCamera> trackballStyle;
    //interactor->SetInteractorStyle(trackballStyle);
    vtkNew<CustomTrackballStyle> customTrackballStyle;
    interactor->SetInteractorStyle(customTrackballStyle);
    interactor->SetRenderWindow(renderWindow);
    interactor->Initialize();

    VisualDebugging::Initialize(renderer);

    MaximizeVTKWindowOnMonitor(renderWindow, 2);

	{
		auto mesh = ReadPLY("C:\\Resources\\3D\\PLY\\Mesh.ply");

		vtkNew<vtkPolyDataMapper> mapper;
		mapper->SetInputData(mesh);

		vtkNew<vtkActor> actor;
		actor->SetMapper(mapper);

		renderer->AddActor(actor);
	}

    vtkSmartPointer<vtkCallbackCommand> keyPressCallback = vtkSmartPointer<vtkCallbackCommand>::New();
    keyPressCallback->SetCallback(OnKeyPress);
    keyPressCallback->SetClientData(renderer);

    vtkSmartPointer<TimerCallback> timerCallback = vtkSmartPointer<TimerCallback>::New();

    interactor->AddObserver(vtkCommand::TimerEvent, timerCallback);
    int timerId = interactor->CreateRepeatingTimer(16);
    if (timerId < 0) {
        std::cerr << "Error: Timer was not created!" << std::endl;
    }

    interactor->AddObserver(vtkCommand::KeyPressEvent, keyPressCallback);

	vtkCamera* camera = renderer->GetActiveCamera();
	camera->SetParallelProjection(true);
	renderer->ResetCamera();

    renderWindow->Render();
    interactor->Start();

    VisualDebugging::Terminate();

    return 0;
}
