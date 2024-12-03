#include <App/App.h>
#include <App/AppEventHandlers.h>
#include <App/CustomTrackBallStyle.h>
#include <Algorithm/KDTree.h>
#include <Algorithm/Octree.hpp>
#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

set<size_t> selectedIndices;

int depthIndex = 0;


AppEventHandler::AppEventHandler()
{
}

AppEventHandler::~AppEventHandler()
{
}

bool OnKeyPress(App* app)
{
	vtkRenderWindowInteractor* interactor = app->GetInteractor();
	vtkRenderWindow* renderWindow = interactor->GetRenderWindow();
	vtkRenderer* renderer = renderWindow->GetRenderers()->GetFirstRenderer();
	vtkCamera* camera = renderer->GetActiveCamera();

	std::string key = interactor->GetKeySym();
	app->SetKeyState(key, true);

	//printf("%s\n", key.c_str());

	if (key == "r")
	{
		std::cout << "Key 'r' was pressed. Resetting camera." << std::endl;
		renderer->ResetCamera();

		// Reset the camera position, focal point, and view up vector to their defaults
		camera->SetPosition(0, 0, 1);          // Reset position to default
		camera->SetFocalPoint(0, 0, 0);        // Reset focal point to origin
		camera->SetViewUp(0, 1, 0);            // Reset the up vector to default (Y-axis up)

		interactor->Render();
	}
	else if (key == "c")
	{
		// Generate a filename using the current time
		std::string filename = "C:\\Resources\\2D\\Captured\\screenshot_" + Time::DateTime() + ".png";

		// Capture the frame
		vtkNew<vtkWindowToImageFilter> windowToImageFilter;
		windowToImageFilter->SetInput(renderWindow);
		windowToImageFilter->Update();

		// Write the captured frame to a PNG file with the timestamped filename
		vtkNew<vtkPNGWriter> pngWriter;
		pngWriter->SetFileName(filename.c_str());
		pngWriter->SetInputConnection(windowToImageFilter->GetOutputPort());
		pngWriter->Write();

		std::cout << "Screenshot saved as " << filename << std::endl;
	}
	else if (key == "p")
	{
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
	else if (key == "1")
	{
		VisualDebugging::ToggleVisibility("Cubes_0");
	}
	else if (key == "2")
	{
		VisualDebugging::ToggleVisibility("Cubes_1");
	}
	else if (key == "3")
	{
	VisualDebugging::ToggleVisibility("Cubes_2");
	}
	else if (key == "4")
	{
	VisualDebugging::ToggleVisibility("Cubes_3");
	}
	else if (key == "5")
	{
	VisualDebugging::ToggleVisibility("Cubes_4");
	}
	else if (key == "6")
	{
	VisualDebugging::ToggleVisibility("Cubes_5");
	}
	else if (key == "7")
	{
	VisualDebugging::ToggleVisibility("Cubes_6");
	}
	else if (key == "8")
	{
	VisualDebugging::ToggleVisibility("Cubes_7");
	}
	else if (key == "9")
	{
	VisualDebugging::ToggleVisibility("Cubes_8");
	}
	else if (key == "0")
	{
	VisualDebugging::ToggleVisibility("Cubes_9");
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
	else if (key == "quoteleft")
	{
		VisualDebugging::ToggleVisibility("axes");
	}
	else if (key == "BackSpace")
	{
		auto viewMatrix = vtkToEigen(camera->GetViewTransformMatrix());
		Eigen::Matrix4f inverseViewMatrix = viewMatrix.inverse();

		Eigen::Vector3f viewDirection = -inverseViewMatrix.block<3, 1>(0, 2);
		viewDirection.normalize();
		Eigen::Vector3f cameraPosition = inverseViewMatrix.block<3, 1>(0, 3);

		//VisualDebugging::AddLine("ViewDirection", cameraPosition, cameraPosition + viewDirection * 100, Color4::Red);

		Eigen::Vector3f zero = (inverseViewMatrix * Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f)).head<3>();
		Eigen::Vector3f right = (inverseViewMatrix * Eigen::Vector4f(10.0f, 0.0f, 0.0f, 1.0f)).head<3>();
		Eigen::Vector3f up = (inverseViewMatrix * Eigen::Vector4f(0.0f, 10.0f, 0.0f, 1.0f)).head<3>();
		Eigen::Vector3f front = (inverseViewMatrix * Eigen::Vector4f(0.0f, 0.0f, 10.0f, 1.0f)).head<3>();

		VisualDebugging::AddLine("ViewDirection", zero, right, Color4::Red);
		VisualDebugging::AddLine("ViewDirection", zero, up, Color4::Green);
		VisualDebugging::AddLine("ViewDirection", zero, front, Color4::Blue);
	}
	else if (key == "App")
	{
		app->CaptureColorAndDepth("C:\\Resources\\2D\\Captured");
	}
	else if (key == "Insert")
	{
		printf("Camera Distance : %f\n", camera->GetDistance());
		printf("Camera Parallel Scale : %f\n", camera->GetParallelScale());
	}
	else if (key == "Return")
	{
	selectedIndices.clear();
	VD::Clear("NN");
	}
	else if (key == "space")
	{
		//interactor->InvokeEvent(vtkCommand::MouseWheelForwardEvent);
	}

	return true;
}

bool OnKeyRelease(App* app)
{
	vtkRenderWindowInteractor* interactor = app->GetInteractor();
	vtkRenderWindow* renderWindow = interactor->GetRenderWindow();
	vtkRenderer* renderer = renderWindow->GetRenderers()->GetFirstRenderer();
	vtkCamera* camera = renderer->GetActiveCamera();

	std::string key = interactor->GetKeySym();
	app->SetKeyState(key, false);

	return true;
}

bool OnMouseButtonPress(App* app, int button)
{
	if (button == 0)
	{
		//selectedIndices.clear();
		//VD::Clear("NN");
	}

	return true;
}

bool OnMouseButtonRelease(App* app, int button)
{
	vtkRenderWindowInteractor* interactor = app->GetInteractor();
	vtkRenderWindow* renderWindow = interactor->GetRenderWindow();
	vtkRenderer* renderer = renderWindow->GetRenderers()->GetFirstRenderer();
	vtkCamera* camera = renderer->GetActiveCamera();

	int* mousePos = interactor->GetEventPosition();
	int mouseX = mousePos[0];
	int mouseY = mousePos[1];

	int* size = renderWindow->GetSize();
	int screenWidth = size[0];
	int screenHeight = size[1];

	float depth = 0.5f;

	//printf("mouseX : %d, mouseY : %d\n", mouseX, mouseY);

	if (1 == button)
	{
		//auto t = Time::Now();

		//float ndcX = (2.0f * mouseX) / screenWidth - 1.0f;
		//float ndcY = (2.0f * mouseY) / screenHeight - 1.0f;

		//Eigen::Vector4f clipSpacePoint(ndcX, ndcY, depth, 1.0f);

		//auto viewMatrix = vtkToEigen(camera->GetViewTransformMatrix());
		//auto projectionMatrix = vtkToEigen(camera->GetProjectionTransformMatrix((float)screenWidth / (float)screenHeight, -1, 1));

		//Eigen::Matrix4f viewProjectionMatrix = projectionMatrix * viewMatrix;
		//Eigen::Matrix4f inverseVPMatrix = viewProjectionMatrix.inverse();
		//Eigen::Vector4f worldSpacePoint4 = inverseVPMatrix * clipSpacePoint;
		//if (worldSpacePoint4.w() != 0.0f) {
		//	worldSpacePoint4 /= worldSpacePoint4.w();
		//}
		//Eigen::Vector3f worldSpacePoint = worldSpacePoint4.head<3>();

		//Eigen::Matrix4f inverseViewMatrix = viewMatrix.inverse();

		///*Eigen::Vector3f viewDirection = -inverseViewMatrix.block<3, 1>(0, 2);
		//viewDirection.normalize();*/

		//Eigen::Vector3f cameraPosition = inverseViewMatrix.block<3, 1>(0, 3);

		//Eigen::Vector3f rayDirection = worldSpacePoint - cameraPosition;
		//rayDirection.normalize();

		//VD::Clear("ViewDirection");
		//VisualDebugging::AddLine("ViewDirection", cameraPosition, cameraPosition + rayDirection * 1000, Color4::Red);

		//auto octree = (Spatial::Octree*)app->registry["octree"];

		//Spatial::Ray ray(cameraPosition, rayDirection);
		//auto result = octree->searchPointsNearRay(ray, 0.2f);
		//t = Time::End(t, "Picking");

		//float weight = 50.0f;

		//if (0 < result.size())
		//{
		//	int candidateIndex = result[0];
		//	float minScore = FLT_MAX;
		//	for (auto& i : result)
		//	{
		//		auto p = octree->points[i];
		//		float distanceToRay = octree->distanceToRay(ray, p) + 0.001f;
		//		auto distanceToOrigin = ((p - cameraPosition).norm() + 0.001f) / weight;
		//		float distanceScore = distanceToRay * distanceToOrigin;
		//		if (distanceScore < minScore)
		//		{
		//			candidateIndex = i;
		//			minScore = distanceScore;
		//		}
		//		VD::AddSphere("NN", p, { 0.15f, 0.15f, 0.15f }, { 0.0f, 0.0f, 1.0f }, Color4::Blue);
		//	}

		//	{
		//		auto p = octree->points[candidateIndex];
		//		VD::AddSphere("NN", p, { 0.2f, 0.2f, 0.2f }, { 0.0f, 0.0f, 1.0f }, Color4::Red);

		//		auto searchResult = octree->radiusSearch(p, 1.0f);
		//		for (auto& i : searchResult)
		//		{
		//			if (0 == selectedIndices.count(i))
		//			{
		//				selectedIndices.insert(i);
		//				auto rp = octree->points[i];
		//				VD::AddSphere("NN", rp, { 0.15f, 0.15f, 0.15f }, { 0.0f, 0.0f, 1.0f }, Color4::Green);
		//			}
		//		}

		//		camera->SetFocalPoint(p.x(), p.y(), p.z());
		//		renderWindow->Render();
		//	}
		//}

		//auto pi = octree->pickPoint(Spatial::Ray(cameraPosition, rayDirection));
		//if (pi != -1)
		//{
		//	auto p = octree->points[(size_t)pi];
		//	//VD::AddSphere("NN", p, { 0.15f, 0.15f, 0.15f }, { 0.0f, 0.0f, 1.0f }, Color4::Red);

		//	camera->SetFocalPoint(p.x(), p.y(), p.z());
		//	renderWindow->Render();
		//}

		////////////auto kdtree = (Algorithm::KDTree*)app->registry["kdtree"];

		////////////Algorithm::Ray ray = {
		////////////	{ cameraPosition.x(), cameraPosition.y(), cameraPosition.z() },
		////////////	{ rayDirection.x(), rayDirection.y(), rayDirection.z() }
		////////////};

		////////////int k = 30;
		////////////float maxDistance = 1.0f;
		////////////std::vector<unsigned int> knn = kdtree->RayKNearestNeighbors(ray, k, maxDistance);

		////////////VD::Clear("KNN");
		////////////auto points = kdtree->GetPoints();
		////////////printf("knn.size() : %d\n", knn.size());
		////////////for (auto& index : knn)
		////////////{
		////////////	auto x = points[index * 3];
		////////////	auto y = points[index * 3 + 1];
		////////////	auto z = points[index * 3 + 2];

		////////////	//printf("")

		////////////	VD::AddSphere("KNN", { x,y,z }, { 0.5f, 0.5f, 0.5f }, { 0.0f, 0.0f, 1.0f }, Color4::Red);
		////////////}

		////////////t = Time::End(t, "Picking");

		////////////float distanceMin = FLT_MAX;
		////////////float minX = FLT_MAX;
		////////////float minY = FLT_MAX;
		////////////float minZ = FLT_MAX;

		////////////vector<size_t> indices;

		////////////for (size_t i = 0; i < 1398561; i++)
		////////////{
		////////////	auto x = *(points + i * 3);
		////////////	auto y = *(points + i * 3 + 1);
		////////////	auto z = *(points + i * 3 + 2);

		////////////	auto distance = sqrt(Algorithm::RayPointDistanceSquared(ray, points + i * 3));

		////////////	if (distance < 0.5f)
		////////////	{
		////////////		indices.push_back(i);

		////////////		//printf("x : %f, y : %f, z : %f, distance : %f\n", x, y, z, distance);

		////////////		VD::AddSphere("KNN", { x,y,z }, { 0.1f, 0.1f, 0.1f }, { 0.0f, 0.0f, 1.0f }, Color4::Green);
		////////////	}

		////////////	if (distance < distanceMin)
		////////////	{
		////////////		distanceMin = distance;
		////////////		minX = x;
		////////////		minY = y;
		////////////		minZ = z;
		////////////	}
		////////////}

		////////////printf("-=-\n");

		//////////////for (size_t i = 0; i < indices.size(); i++)
		//////////////{
		//////////////	auto x = *(points + indices[i] * 3);
		//////////////	auto y = *(points + indices[i] * 3 + 1);
		//////////////	auto z = *(points + indices[i] * 3 + 2);

		//////////////	auto dx = minX - x;
		//////////////	auto dy = minY - y;
		//////////////	auto dz = minZ - z;

		//////////////	float distance = sqrt(dx * dx + dy * dy + dz * dz);

		//////////////	//printf("x : %f, y : %f, z : %f, distance : %f\n", x, y, z, distance);

		//////////////	if (distance > 0.5f)
		//////////////	{
		//////////////		VD::AddSphere("KNN", { x,y,z }, { 0.1f, 0.1f, 0.1f }, { 0.0f, 0.0f, 1.0f }, Color4::Yellow);
		//////////////	}
		//////////////}

		////////////VD::AddSphere("KNN", { minX, minY, minZ }, { 0.1f,0.1f,0.1f }, { 0.0f, 0.0f, 1.0f }, Color4::Blue);

		////////////camera->SetFocalPoint(minX, minY, minZ);
		////////////renderWindow->Render();
	}

	return true;
}

bool OnMouseMove(App* app, int posx, int posy, int lastx, int lasty, bool lButton, bool mButton, bool rButton)
{
	vtkRenderWindowInteractor* interactor = app->GetInteractor();
	vtkRenderWindow* renderWindow = interactor->GetRenderWindow();
	vtkRenderer* renderer = renderWindow->GetRenderers()->GetFirstRenderer();
	vtkCamera* camera = renderer->GetActiveCamera();

	if (lButton)
	{
		//int* mousePos = interactor->GetEventPosition();
		//int mouseX = mousePos[0];
		//int mouseY = mousePos[1];

		//int* size = renderWindow->GetSize();
		//int screenWidth = size[0];
		//int screenHeight = size[1];

		//float depth = 0.5f;
		////depth = 1.0f;

		//auto t = Time::Now();

		//float ndcX = (2.0f * mouseX) / screenWidth - 1.0f;
		//float ndcY = (2.0f * mouseY) / screenHeight - 1.0f;

		//Eigen::Vector4f clipSpacePoint(ndcX, ndcY, depth, 1.0f);

		//auto viewMatrix = vtkToEigen(camera->GetViewTransformMatrix());
		//auto projectionMatrix = vtkToEigen(camera->GetProjectionTransformMatrix((float)screenWidth / (float)screenHeight, -1, 1));

		//Eigen::Matrix4f viewProjectionMatrix = projectionMatrix * viewMatrix;
		//Eigen::Matrix4f inverseVPMatrix = viewProjectionMatrix.inverse();
		//Eigen::Vector4f worldSpacePoint4 = inverseVPMatrix * clipSpacePoint;
		//if (worldSpacePoint4.w() != 0.0f) {
		//	worldSpacePoint4 /= worldSpacePoint4.w();
		//}
		//Eigen::Vector3f worldSpacePoint = worldSpacePoint4.head<3>();

		//Eigen::Matrix4f inverseViewMatrix = viewMatrix.inverse();

		///*Eigen::Vector3f viewDirection = -inverseViewMatrix.block<3, 1>(0, 2);
		//viewDirection.normalize();*/

		//Eigen::Vector3f cameraPosition = inverseViewMatrix.block<3, 1>(0, 3);

		//Eigen::Vector3f rayDirection = worldSpacePoint - cameraPosition;
		//rayDirection.normalize();

		////VD::Clear("ViewDirection");
		////VisualDebugging::AddLine("ViewDirection", cameraPosition, cameraPosition + rayDirection * 1000, Color4::Red);

		////VD::Clear("NN");
		//auto octree = (Spatial::Octree*)app->registry["octree"];

		//Spatial::Ray ray(cameraPosition, rayDirection);
		//auto result = octree->searchPointsNearRay(ray, 0.2f);
		//t = Time::End(t, "Picking");

		//float weight = 2.0f;

		//static Eigen::Vector3f lastPoint(FLT_MAX, FLT_MAX, FLT_MAX);

		//if (0 < result.size())
		//{
		//	int candidateIndex = result[0];
		//	float minScore = FLT_MAX;
		//	for (auto& i : result)
		//	{
		//		auto p = octree->points[i];
		//		float distanceToRay = octree->distanceToRay(ray, p) + 0.001f;
		//		auto distanceToOrigin = ((p - cameraPosition).norm() + 0.001f) / weight;
		//		float distanceToLastPoint = ((p - lastPoint).norm() + 0.001f) / weight;
		//		float distanceScore = distanceToRay * distanceToOrigin * distanceToLastPoint;
		//		if (distanceScore < minScore)
		//		{
		//			candidateIndex = i;
		//			minScore = distanceScore;
		//		}
		//		VD::AddSphere("NN", p, { 0.15f, 0.15f, 0.15f }, { 0.0f, 0.0f, 1.0f }, Color4::Blue);
		//	}

		//	{
		//		auto p = octree->points[candidateIndex];
		//		VD::AddSphere("NN", p, { 0.2f, 0.2f, 0.2f }, { 0.0f, 0.0f, 1.0f }, Color4::Red);

		//		auto searchResult = octree->radiusSearch(p, 1.0f);
		//		for (auto& i : searchResult)
		//		{
		//			auto rp = octree->points[i];
		//			lastPoint = rp;
		//			VD::AddSphere("NN", rp, { 0.15f, 0.15f, 0.15f }, { 0.0f, 0.0f, 1.0f }, Color4::Green);
		//		}
		//	}
		//}
	}

	//vtkRenderWindowInteractor* interactor = app->GetInteractor();
	//vtkRenderWindow* renderWindow = interactor->GetRenderWindow();
	//vtkRenderer* renderer = renderWindow->GetRenderers()->GetFirstRenderer();
	//vtkCamera* camera = renderer->GetActiveCamera();

	//int* mousePos = interactor->GetEventPosition();
	//int mouseX = mousePos[0];
	//int mouseY = mousePos[1];

	//int* size = renderWindow->GetSize();
	//int screenWidth = size[0];
	//int screenHeight = size[1];

	//float depth = 0.5f;
	//depth = 1.0f;

	//printf("mouseX : %d, mouseY : %d\n", mouseX, mouseY);

	//if (lButton)
	//{
	//	float ndcX = (2.0f * mouseX) / screenWidth - 1.0f;
	//	float ndcY = (2.0f * mouseY) / screenHeight - 1.0f;

	//	Eigen::Vector4f clipSpacePoint(ndcX, ndcY, depth, 1.0f);

	//	auto viewMatrix = vtkToEigen(camera->GetViewTransformMatrix());
	//	auto projectionMatrix = vtkToEigen(camera->GetProjectionTransformMatrix((float)screenWidth / (float)screenHeight, -1, 1));

	//	Eigen::Matrix4f viewProjectionMatrix = projectionMatrix * viewMatrix;
	//	Eigen::Matrix4f inverseVPMatrix = viewProjectionMatrix.inverse();
	//	Eigen::Vector4f worldSpacePoint4 = inverseVPMatrix * clipSpacePoint;
	//	if (worldSpacePoint4.w() != 0.0f) {
	//		worldSpacePoint4 /= worldSpacePoint4.w();
	//	}
	//	Eigen::Vector3f worldSpacePoint = worldSpacePoint4.head<3>();

	//	Eigen::Matrix4f inverseViewMatrix = viewMatrix.inverse();

	//	/*Eigen::Vector3f viewDirection = -inverseViewMatrix.block<3, 1>(0, 2);
	//	viewDirection.normalize();*/

	//	Eigen::Vector3f cameraPosition = inverseViewMatrix.block<3, 1>(0, 3);

	//	Eigen::Vector3f rayDirection = worldSpacePoint - cameraPosition;
	//	rayDirection.normalize();

	//	//VisualDebugging::AddLine("ViewDirection", cameraPosition, cameraPosition + rayDirection * 1000, Color4::Red);
	//	VisualDebugging::AddLine("ViewDirection", cameraPosition, worldSpacePoint, Color4::Red);
	//}

	return true;
}

bool OnMouseWheelScroll(App* app, bool isForward)
{
	vtkRenderWindowInteractor* interactor = app->GetInteractor();
	CustomTrackballStyle* interactorStyle = (CustomTrackballStyle*)interactor->GetInteractorStyle();
	vtkRenderWindow* renderWindow = interactor->GetRenderWindow();
	vtkRenderer* renderer = renderWindow->GetRenderers()->GetFirstRenderer();
	vtkCamera* camera = renderer->GetActiveCamera();

	if (app->GetKeyState("Control_L"))
	{
		// Get the current clipping range of the camera
		double* range = camera->GetClippingRange();

		// Adjust clipping range based on scroll direction
		if (isForward)
		{
			printf("Forward\n");
			range[0] *= 1.01;  // Increase near clipping plane distance
		}
		else
		{
			printf("Backward\n");
			range[0] *= 0.99;  // Decrease near clipping plane distance
		}

		camera->SetClippingRange(range[0], range[1]);

		// Set the modified clipping range back to the camera
		//camera->SetClippingRange(range);

		// You can verify the values
		double* clippingRange = camera->GetClippingRange();
		std::cout << "Near clipping distance: " << clippingRange[0] << std::endl;
		std::cout << "Far clipping distance: " << clippingRange[1] << std::endl;

		//renderWindow->Render();
		
		return false;
	}

	return true;
}

#ifdef _WINDOWS
bool OnUSBEvent(App* app, USBEvent usbEvent)
{
	auto interactor = app->GetInteractor();

	//printf("%02X %02X %02X - %02X %02X %02X\n", a, b, c, d, e, f);
	if (0x11 == usbEvent.data[0] && 0x0A == usbEvent.data[1]) // Big Wheel
	{
		if (0x01 == usbEvent.data[2]) // Right
		{
			interactor->InvokeEvent(vtkCommand::MouseWheelForwardEvent);
		}
		else if (0x02 == usbEvent.data[2]) // Left
		{
			interactor->InvokeEvent(vtkCommand::MouseWheelBackwardEvent);
		}
	}

	return true;
}
#endif
