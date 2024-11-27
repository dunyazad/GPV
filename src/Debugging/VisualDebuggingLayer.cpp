#include <Debugging/VisualDebuggingLayer.h>
#include <Debugging/vtkPolygonalFrustumSource.h>

#include <Algorithm/CustomPolyDataFilter.h>

VisualDebuggingLayer::VisualDebuggingLayer(const string& layerName)
	: layerName(layerName) {}

VisualDebuggingLayer::~VisualDebuggingLayer() {}

void VisualDebuggingLayer::Initialize(vtkSmartPointer<vtkRenderer> renderer)
{
	this->renderer = renderer;
	renderWindow = renderer->GetRenderWindow();

#pragma region Point
	{
		pointPolyData = vtkSmartPointer<vtkPolyData>::New();
		pointPolyDataMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
		pointPolyDataMapper->SetInputData(pointPolyData);
		pointPolyDataMapper->SetScalarModeToUsePointData();
		pointActor = vtkSmartPointer<vtkActor>::New();
		pointActor->SetMapper(linePolyDataMapper);
		//pointActor->SetObjectName(layerName + ".pointActor");

		vtkNew<vtkPoints> points;
		pointPolyData->SetPoints(points);

		vtkNew<vtkUnsignedCharArray> colors;
		colors->SetNumberOfComponents(3);
		pointPolyData->GetCellData()->SetScalars(colors);

		renderer->AddActor(pointActor);
	}
#pragma endregion

#pragma region Line
	{
		linePolyData = vtkSmartPointer<vtkPolyData>::New();
		linePolyDataMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
		linePolyDataMapper->SetInputData(linePolyData);
		linePolyDataMapper->SetScalarModeToUsePointData();
		lineActor = vtkSmartPointer<vtkActor>::New();
		lineActor->SetMapper(linePolyDataMapper);
		//lineActor->GetProperty()->SetLineWidth(5);
		//lineActor->GetProperty()->LightingOff();
		//lineActor->SetObjectName(layerName + ".lineActor");

		vtkNew<vtkPoints> points;
		linePolyData->SetPoints(points);

		vtkNew<vtkCellArray> lines;
		linePolyData->SetLines(lines);

		vtkNew<vtkUnsignedCharArray> colors;
		colors->SetNumberOfComponents(3);
		linePolyData->GetCellData()->SetScalars(colors);

		renderer->AddActor(lineActor);
	}
#pragma endregion

#pragma region Triangle
	{
		trianglePolyData = vtkSmartPointer<vtkPolyData>::New();
		trianglePolyDataMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
		trianglePolyDataMapper->SetInputData(trianglePolyData);
		trianglePolyDataMapper->SetScalarModeToUsePointData();
		triangleActor = vtkSmartPointer<vtkActor>::New();
		triangleActor->SetMapper(trianglePolyDataMapper);
		//triangleActor->SetObjectName(layerName + ".triangleActor");

		vtkNew<vtkPoints> points;
		trianglePolyData->SetPoints(points);

		vtkNew<vtkCellArray> triangles;
		trianglePolyData->SetPolys(triangles);

		vtkNew<vtkUnsignedCharArray> colors;
		colors->SetNumberOfComponents(3);
		trianglePolyData->GetCellData()->SetScalars(colors);

		renderer->AddActor(triangleActor);
	}
#pragma endregion

#pragma region Sphere
	{
		spherePolyData = vtkSmartPointer<vtkPolyData>::New();

		vtkNew<vtkPoints> points;
		spherePolyData->SetPoints(points);

		vtkNew<vtkUnsignedCharArray> colors;
		colors->SetName("Colors");
		colors->SetNumberOfComponents(3);
		spherePolyData->GetPointData()->AddArray(colors);

		vtkNew<vtkDoubleArray> scales;
		scales->SetName("Scales");
		scales->SetNumberOfComponents(3);
		spherePolyData->GetPointData()->AddArray(scales);

		vtkNew<vtkDoubleArray> normals;
		normals->SetNumberOfComponents(3);
		normals->SetName("Normals");
		spherePolyData->GetPointData()->AddArray(normals);

		vtkNew<vtkSphereSource> sphereSource;
		sphereSource->Update();

		//sphereGlyph3D = vtkSmartPointer<vtkGlyph3D>::New();
		spherePolyDataMapper = vtkSmartPointer<vtkGlyph3DMapper>::New();
		spherePolyDataMapper->SetSourceConnection(sphereSource->GetOutputPort());
		spherePolyDataMapper->SetInputData(spherePolyData);
		spherePolyDataMapper->SetScalarModeToUsePointFieldData();
		spherePolyDataMapper->SetScaleModeToScaleByVectorComponents();
		spherePolyDataMapper->SetScaleArray("Scales");
		spherePolyDataMapper->SelectColorArray("Colors");
		spherePolyDataMapper->SetOrientationArray("Normals");
		spherePolyDataMapper->OrientOn();
		spherePolyDataMapper->Update();

		sphereActor = vtkSmartPointer<vtkActor>::New();
		sphereActor->SetMapper(spherePolyDataMapper);
		// sphereActor->GetProperty()->SetAmbient(1.0);
		// sphereActor->GetProperty()->SetDiffuse(0.0);
		//sphereActor->SetObjectName(layerName + ".sphereActor");

		renderer->AddActor(sphereActor);
	}
#pragma endregion

#pragma region Cube
	{
		cubePolyData = vtkSmartPointer<vtkPolyData>::New();

		vtkNew<vtkPoints> points;
		cubePolyData->SetPoints(points);

		vtkNew<vtkUnsignedCharArray> colors;
		colors->SetName("Colors");
		colors->SetNumberOfComponents(3);
		cubePolyData->GetPointData()->AddArray(colors);

		vtkNew<vtkDoubleArray> scales;
		scales->SetNumberOfComponents(1);
		scales->SetName("Scales");
		scales->SetNumberOfComponents(3);
		cubePolyData->GetPointData()->AddArray(scales);

		vtkNew<vtkDoubleArray> normals;
		normals->SetNumberOfComponents(3);
		normals->SetName("Normals");
		cubePolyData->GetPointData()->AddArray(normals);

		vtkNew<vtkCubeSource> cubeSource;
		cubeSource->Update();

		//cubeGlyph3D = vtkSmartPointer<vtkGlyph3D>::New();
		cubePolyDataMapper = vtkSmartPointer<vtkGlyph3DMapper>::New();
		cubePolyDataMapper->SetSourceConnection(cubeSource->GetOutputPort());
		cubePolyDataMapper->SetInputData(cubePolyData);
		cubePolyDataMapper->SetScalarModeToUsePointFieldData();
		cubePolyDataMapper->SetScaleModeToScaleByVectorComponents();
		cubePolyDataMapper->SetScaleArray("Scales");
		cubePolyDataMapper->SelectColorArray("Colors");
		cubePolyDataMapper->SetOrientationArray("Normals");
		cubePolyDataMapper->OrientOn();
		cubePolyDataMapper->Update();

		cubeActor = vtkSmartPointer<vtkActor>::New();
		cubeActor->SetMapper(cubePolyDataMapper);
		// cubeActor->GetProperty()->SetAmbient(1.0);
		// cubeActor->GetProperty()->SetDiffuse(0.0);
		//cubeActor->SetObjectName(layerName + ".cubeActor");

		renderer->AddActor(cubeActor);
	}
#pragma endregion

#pragma region Glyph
	{
		glyphPolyData = vtkSmartPointer<vtkPolyData>::New();

		vtkNew<vtkPoints> points;
		glyphPolyData->SetPoints(points);

		vtkNew<vtkUnsignedCharArray> colors;
		colors->SetName("Colors");
		colors->SetNumberOfComponents(3);
		glyphPolyData->GetPointData()->AddArray(colors);

		vtkNew<vtkDoubleArray> scales;
		scales->SetNumberOfComponents(1);
		scales->SetName("Scales");
		scales->SetNumberOfComponents(3);
		glyphPolyData->GetPointData()->AddArray(scales);

		vtkNew<vtkDoubleArray> normals;
		normals->SetNumberOfComponents(3);
		normals->SetName("Normals");
		glyphPolyData->GetPointData()->AddArray(normals);

		vtkNew<vtkPolygonalFrustumSource> glyphSource;
		glyphSource->SetNumberOfSides(32);
		glyphSource->SetTopRadius(1.0);
		glyphSource->SetBottomRadius(0.5);
		glyphSource->SetHeight(4.0);
		glyphSource->Update();

		glyphPolyDataMapper = vtkSmartPointer<vtkGlyph3DMapper>::New();
		glyphPolyDataMapper->SetSourceConnection(glyphSource->GetOutputPort());
		glyphPolyDataMapper->SetInputData(glyphPolyData);
		glyphPolyDataMapper->SetScalarModeToUsePointFieldData();
		glyphPolyDataMapper->SetScaleModeToScaleByVectorComponents();
		glyphPolyDataMapper->SetScaleArray("Scales");
		glyphPolyDataMapper->SelectColorArray("Colors");
		glyphPolyDataMapper->SetOrientationArray("Normals");
		glyphPolyDataMapper->OrientOn();
		glyphPolyDataMapper->Update();

		glyphActor = vtkSmartPointer<vtkActor>::New();
		glyphActor->SetMapper(glyphPolyDataMapper);
		// glyphActor->GetProperty()->SetAmbient(1.0);
		// glyphActor->GetProperty()->SetDiffuse(0.0);
		//glyphActor->SetObjectName(layerName + ".glyphActor");

		renderer->AddActor(glyphActor);
	}
#pragma endregion

#pragma region Arrow
	{
		arrowPolyData = vtkSmartPointer<vtkPolyData>::New();

		vtkNew<vtkPoints> points;
		arrowPolyData->SetPoints(points);

		vtkNew<vtkDoubleArray> scales;
		scales->SetNumberOfComponents(3);
		scales->SetName("Scales");
		arrowPolyData->GetPointData()->AddArray(scales);

		vtkNew<vtkDoubleArray> normals;
		normals->SetNumberOfComponents(3);
		normals->SetName("Normals");
		arrowPolyData->GetPointData()->AddArray(normals);

		vtkNew<vtkUnsignedCharArray> colors;
		colors->SetName("Colors");
		colors->SetNumberOfComponents(3);
		arrowPolyData->GetPointData()->AddArray(colors);

		vtkNew<vtkArrowSource> arrowSource;
		arrowSource->Update();

		arrowGlyph3D = vtkSmartPointer<vtkGlyph3D>::New();
		arrowGlyph3D->SetSourceConnection(arrowSource->GetOutputPort());
		arrowGlyph3D->SetInputData(arrowPolyData);
		arrowGlyph3D->SetScaleModeToScaleByScalar();
		arrowGlyph3D->SetColorModeToColorByScalar();

		arrowGlyph3D->SetInputArrayToProcess(
			0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS, "Scales");
		arrowGlyph3D->SetInputArrayToProcess(
			1, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS, "Normals");
		arrowGlyph3D->SetInputArrayToProcess(
			3, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS, "Colors");
		arrowGlyph3D->Update();

		arrowPolyDataMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
		arrowPolyDataMapper->SetInputConnection(arrowGlyph3D->GetOutputPort());

		arrowActor = vtkSmartPointer<vtkActor>::New();
		arrowActor->SetMapper(arrowPolyDataMapper);
		// arrowActor->GetProperty()->SetAmbient(1.0);
		// arrowActor->GetProperty()->SetDiffuse(0.0);
		//arrowActor->SetObjectName(layerName + ".arrowActor");

		renderer->AddActor(arrowActor);
	}
#pragma endregion

#pragma region wiredBox
	{
		wiredBoxPolyData = vtkSmartPointer<vtkPolyData>::New();

		vtkNew<vtkPoints> points;
		wiredBoxPolyData->SetPoints(points);

		vtkNew<vtkDoubleArray> scales;
		scales->SetNumberOfComponents(3);
		scales->SetName("Scales");
		wiredBoxPolyData->GetPointData()->AddArray(scales);

		vtkNew<vtkDoubleArray> normals;
		normals->SetNumberOfComponents(3);
		normals->SetName("Normals");
		wiredBoxPolyData->GetPointData()->AddArray(normals);

		vtkNew<vtkUnsignedCharArray> colors;
		colors->SetName("Colors");
		colors->SetNumberOfComponents(3);
		wiredBoxPolyData->GetPointData()->AddArray(colors);

		vtkNew<vtkPolyData> wiredBox;
		vtkNew<vtkPoints> wiredBoxPoints;
		vtkNew<vtkCellArray> wiredBoxLines;

		auto pi0 = wiredBoxPoints->InsertNextPoint(-0.5f, -0.5f, -0.5f);
		auto pi1 = wiredBoxPoints->InsertNextPoint(0.5f, -0.5f, -0.5f);
		auto pi2 = wiredBoxPoints->InsertNextPoint(-0.5f, 0.5f, -0.5f);
		auto pi3 = wiredBoxPoints->InsertNextPoint(0.5f, 0.5f, -0.5f);
		auto pi4 = wiredBoxPoints->InsertNextPoint(-0.5f, -0.5f, 0.5f);
		auto pi5 = wiredBoxPoints->InsertNextPoint(0.5f, -0.5f, 0.5f);
		auto pi6 = wiredBoxPoints->InsertNextPoint(-0.5f, 0.5f, 0.5f);
		auto pi7 = wiredBoxPoints->InsertNextPoint(0.5f, 0.5f, 0.5f);

		vtkIdType pids[][2] = {
			{pi0, pi1}, {pi1, pi3}, {pi3, pi2}, {pi2, pi0},
			{pi4, pi5}, {pi5, pi7}, {pi7, pi6}, {pi6, pi4},
			{pi0, pi4}, {pi1, pi5}, {pi2, pi6}, {pi3, pi7}
		};
		wiredBoxLines->InsertNextCell(2, pids[0]);
		wiredBoxLines->InsertNextCell(2, pids[1]);
		wiredBoxLines->InsertNextCell(2, pids[2]);
		wiredBoxLines->InsertNextCell(2, pids[3]);
		wiredBoxLines->InsertNextCell(2, pids[4]);
		wiredBoxLines->InsertNextCell(2, pids[5]);
		wiredBoxLines->InsertNextCell(2, pids[6]);
		wiredBoxLines->InsertNextCell(2, pids[7]);
		wiredBoxLines->InsertNextCell(2, pids[8]);
		wiredBoxLines->InsertNextCell(2, pids[9]);
		wiredBoxLines->InsertNextCell(2, pids[10]);
		wiredBoxLines->InsertNextCell(2, pids[11]);

		wiredBox->SetPoints(wiredBoxPoints);
		wiredBox->SetLines(wiredBoxLines);
		wiredBox->Modified();

		vtkNew<CustomPolyDataFilter> wiredBoxSource;
		wiredBoxSource->SetInputData(wiredBox);
		wiredBoxSource->Update();

		vtkNew<vtkCubeSource> cubeSource;
		cubeSource->Update();

		//vtkNew<vtkBoxsour

		wiredBoxGlyph3D = vtkSmartPointer<vtkGlyph3D>::New();
		//wiredBoxGlyph3D->SetSourceConnection(wiredBoxSource->GetOutputPort());
		wiredBoxGlyph3D->SetSourceConnection(cubeSource->GetOutputPort());
		wiredBoxGlyph3D->SetInputData(wiredBoxPolyData);
		wiredBoxGlyph3D->SetScaleModeToScaleByScalar();
		//wiredBoxGlyph3D->SetVectorModeToUseVector();
		//wiredBoxGlyph3D->SetScaleModeToScaleByVectorComponents();
		wiredBoxGlyph3D->SetColorModeToColorByScalar();

		wiredBoxGlyph3D->SetInputArrayToProcess(
			0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS, "Scales");
		wiredBoxGlyph3D->SetInputArrayToProcess(
			1, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS, "Normals");
		wiredBoxGlyph3D->SetInputArrayToProcess(
			3, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS, "Colors");
		wiredBoxGlyph3D->Update();

		wiredBoxPolyDataMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
		wiredBoxPolyDataMapper->SetInputConnection(wiredBoxGlyph3D->GetOutputPort());

		wiredBoxActor = vtkSmartPointer<vtkActor>::New();
		wiredBoxActor->SetMapper(wiredBoxPolyDataMapper);
		// wiredBoxActor->GetProperty()->SetAmbient(1.0);
		// wiredBoxActor->GetProperty()->SetDiffuse(0.0);
		//wiredBoxActor->SetObjectName(layerName + ".wiredBoxActor");

		renderer->AddActor(wiredBoxActor);
	}
#pragma endregion
}

void VisualDebuggingLayer::Terminate()
{
#pragma region Point
	if (nullptr != pointPolyData)
	{
		pointPolyData = nullptr;
	}

	if (nullptr != pointPolyDataMapper)
	{
		pointPolyDataMapper = nullptr;
	}

	if (nullptr != pointActor)
	{
		renderer->RemoveActor(pointActor);
		pointActor = nullptr;
	}
#pragma endregion

#pragma region Line
	if (nullptr != linePolyData)
	{
		linePolyData = nullptr;
	}

	if (nullptr != linePolyDataMapper)
	{
		linePolyDataMapper = nullptr;
	}

	if (nullptr != lineActor)
	{
		renderer->RemoveActor(lineActor);
		lineActor = nullptr;
	}
#pragma endregion

#pragma region Triangle
	if (nullptr != trianglePolyData)
	{
		trianglePolyData = nullptr;
	}

	if (nullptr != trianglePolyDataMapper)
	{
		trianglePolyDataMapper = nullptr;
	}

	if (nullptr != triangleActor)
	{
		renderer->RemoveActor(triangleActor);
		triangleActor = nullptr;
	}
#pragma endregion

#pragma region Sphere
	if (nullptr != spherePolyData)
	{
		spherePolyData = nullptr;
	}

	if (nullptr != spherePolyDataMapper)
	{
		spherePolyDataMapper = nullptr;
	}

	if (nullptr != sphereActor)
	{
		renderer->RemoveActor(sphereActor);
		sphereActor = nullptr;
	}
#pragma endregion

#pragma region Cube
	if (nullptr != cubePolyData)
	{
		cubePolyData = nullptr;
	}

	if (nullptr != cubePolyDataMapper)
	{
		cubePolyDataMapper = nullptr;
	}

	if (nullptr != cubeActor)
	{
		renderer->RemoveActor(cubeActor);
		cubeActor = nullptr;
	}
#pragma endregion

#pragma region Glyph
	if (nullptr != glyphPolyData)
	{
		glyphPolyData = nullptr;
	}

	if (nullptr != glyphPolyDataMapper)
	{
		glyphPolyDataMapper = nullptr;
	}

	if (nullptr != glyphActor)
	{
		renderer->RemoveActor(glyphActor);
		glyphActor = nullptr;
	}
#pragma endregion

#pragma region Arrow
	if (nullptr != arrowGlyph3D)
	{
		arrowGlyph3D = nullptr;
	}

	if (nullptr != arrowPolyData)
	{
		arrowPolyData = nullptr;
	}

	if (nullptr != arrowPolyDataMapper)
	{
		arrowPolyDataMapper = nullptr;
	}

	if (nullptr != arrowActor)
	{
		renderer->RemoveActor(arrowActor);
		arrowActor = nullptr;
	}
#pragma endregion

#pragma region WiredBox
	if (nullptr != wiredBoxGlyph3D)
	{
		wiredBoxGlyph3D = nullptr;
	}

	if (nullptr != wiredBoxPolyData)
	{
		wiredBoxPolyData = nullptr;
	}

	if (nullptr != wiredBoxPolyDataMapper)
	{
		wiredBoxPolyDataMapper = nullptr;
	}

	if (nullptr != wiredBoxActor)
	{
		renderer->RemoveActor(wiredBoxActor);
		wiredBoxActor = nullptr;
	}
#pragma endregion
}

void VisualDebuggingLayer::Update()
{
	DrawPoints();
	DrawLines();
	DrawTriangle();
	DrawSpheres();
	DrawCubes();
	DrawGlyphs();
	DrawArrows();
	DrawWiredBoxes();

	// renderWindow->Render();
}

void VisualDebuggingLayer::Clear()
{
	map<string, Representation> representations;
	map<string, bool> visibilities;

	Terminate();
	Initialize(renderer);
}

void VisualDebuggingLayer::AddPoint(const Eigen::Vector3f& p, const Color4& color)
{
	pointInfosToDraw.push_back(std::make_tuple(p, color));
}

void VisualDebuggingLayer::AddLine(const Eigen::Vector3f& p0, const Eigen::Vector3f& p1, const Color4& color)
{
	lineInfosToDraw.push_back(std::make_tuple(p0, p1, color));
}

void VisualDebuggingLayer::AddTriangle(const Eigen::Vector3f& p0, const Eigen::Vector3f& p1,
	const Eigen::Vector3f& p2, const Color4& color)
{
	triangleInfosToDraw.push_back(std::make_tuple(p0, p1, p2, color));
}

void VisualDebuggingLayer::AddSphere(const Eigen::Vector3f& center, const Eigen::Vector3f& scale, const Eigen::Vector3f& normal, const Color4& color)
{
	sphereInfosToDraw.push_back(std::make_tuple(center, scale, normal, color));
}

void VisualDebuggingLayer::AddCube(const Eigen::Vector3f& center, const Eigen::Vector3f& scale, const Eigen::Vector3f& normal, const Color4& color)
{
	cubeInfosToDraw.push_back(std::make_tuple(center, scale, normal, color));
}

void VisualDebuggingLayer::AddGlyph(const Eigen::Vector3f& center, const Eigen::Vector3f& scale, const Eigen::Vector3f& normal, const Color4& color)
{
	glyphInfosToDraw.push_back(std::make_tuple(center, scale, normal, color));
}

void VisualDebuggingLayer::AddArrow(const Eigen::Vector3f& center, const Eigen::Vector3f& normal, float scale, const Color4& color)
{
	arrowInfosToDraw.push_back(std::make_tuple(center, normal, scale, color));
}

void VisualDebuggingLayer::AddWiredBox(const Eigen::Vector3f& boxMin, const Eigen::Vector3f& boxMax, const Color4& color)
{
	Eigen::Vector3f center = (boxMin + boxMax) * 0.5f;
	Eigen::Vector3f scale = boxMax - boxMin;
	wiredBoxInfosToDraw.push_back(std::make_tuple(center, scale, color));
}

void VisualDebuggingLayer::ShowAll(bool show)
{
	ShowPoints(show);
	ShowLines(show);
	ShowTriangles(show);
	ShowSpheres(show);
	ShowCubes(show);
	ShowArrows(show);
	ShowWiredBoxes(show);
}

void VisualDebuggingLayer::ToggleVisibilityAll()
{
	TogglePoints();
	ToggleLines();
	ToggleTriangles();
	ToggleSpheres();
	ToggleCubes();
	ToggleArrows();
	ToggleWiredBoxes();
}

void VisualDebuggingLayer::SetRepresentationAll(Representation representation)
{
	SetRepresentationPoints(representation);
	SetRepresentationLines(representation);
	SetRepresentationTriangles(representation);
	SetRepresentationSpheres(representation);
	SetRepresentationCubes(representation);
	SetRepresentationArrows(representation);
	SetRepresentationWiredBoxes(representation);
}

void VisualDebuggingLayer::ToggleAllRepresentation()
{
	TogglePointsRepresentation();
	ToggleLinesRepresentation();
	ToggleTrianglesRepresentation();
	ToggleSpheresRepresentation();
	ToggleCubesRepresentation();
	ToggleArrowsRepresentation();
	ToggleWiredBoxesRepresentation();
}

void VisualDebuggingLayer::ShowPoints(bool show)
{
	if (nullptr != pointActor)
	{
		ShowActor(renderer, pointActor, show);
	}
}

void VisualDebuggingLayer::TogglePoints()
{
	if (nullptr != pointActor)
	{
		ToggleActorVisibility(renderer, pointActor);
	}
}

void VisualDebuggingLayer::SetRepresentationPoints(Representation representation)
{
	if (nullptr != pointActor)
	{
		SetActorRepresentation(renderer, pointActor, representation);
	}
}

void VisualDebuggingLayer::TogglePointsRepresentation()
{
	if (nullptr != pointActor)
	{
		ToggleActorRepresentation(renderer, pointActor);
	}
}

void VisualDebuggingLayer::ShowLines(bool show)
{
	if (nullptr != lineActor)
	{
		ShowActor(renderer, lineActor, show);
	}
}

void VisualDebuggingLayer::ToggleLines()
{
	if (nullptr != lineActor)
	{
		ToggleActorVisibility(renderer, lineActor);
	}
}

void VisualDebuggingLayer::SetRepresentationLines(Representation representation)
{
	if (nullptr != lineActor)
	{
		SetActorRepresentation(renderer, lineActor, representation);
	}
}

void VisualDebuggingLayer::ToggleLinesRepresentation()
{
	if (nullptr != lineActor)
	{
		ToggleActorRepresentation(renderer, lineActor);
	}
}

void VisualDebuggingLayer::ShowTriangles(bool show)
{
	if (nullptr != triangleActor)
	{
		ShowActor(renderer, triangleActor, show);
	}
}

void VisualDebuggingLayer::ToggleTriangles()
{
	if (nullptr != triangleActor)
	{
		ToggleActorVisibility(renderer, triangleActor);
	}
}

void VisualDebuggingLayer::SetRepresentationTriangles(Representation representation)
{
	if (nullptr != triangleActor)
	{
		SetActorRepresentation(renderer, triangleActor, representation);
	}
}

void VisualDebuggingLayer::ToggleTrianglesRepresentation()
{
	if (nullptr != triangleActor)
	{
		ToggleActorRepresentation(renderer, triangleActor);
	}
}

void VisualDebuggingLayer::ShowSpheres(bool show)
{
	if (nullptr != sphereActor)
	{
		ShowActor(renderer, sphereActor, show);
	}
}

void VisualDebuggingLayer::ToggleSpheres()
{
	if (nullptr != sphereActor)
	{
		ToggleActorVisibility(renderer, sphereActor);
	}
}

void VisualDebuggingLayer::SetRepresentationSpheres(Representation representation)
{
	if (nullptr != sphereActor)
	{
		SetActorRepresentation(renderer, sphereActor, representation);
	}
}

void VisualDebuggingLayer::ToggleSpheresRepresentation()
{
	if (nullptr != sphereActor)
	{
		ToggleActorRepresentation(renderer, sphereActor);
	}
}

void VisualDebuggingLayer::ShowCubes(bool show)
{
	if (nullptr != cubeActor)
	{
		ShowActor(renderer, cubeActor, show);
	}
}

void VisualDebuggingLayer::ToggleCubes()
{
	if (nullptr != cubeActor)
	{
		ToggleActorVisibility(renderer, cubeActor);
	}
}

void VisualDebuggingLayer::SetRepresentationCubes(Representation representation)
{
	if (nullptr != cubeActor)
	{
		SetActorRepresentation(renderer, cubeActor, representation);
	}
}

void VisualDebuggingLayer::ToggleCubesRepresentation()
{
	if (nullptr != cubeActor)
	{
		ToggleActorRepresentation(renderer, cubeActor);
	}
}

void VisualDebuggingLayer::ShowArrows(bool show)
{
	if (nullptr != arrowActor)
	{
		ShowActor(renderer, arrowActor, show);
	}
}

void VisualDebuggingLayer::ToggleArrows()
{
	if (nullptr != arrowActor)
	{
		ToggleActorVisibility(renderer, arrowActor);
	}
}

void VisualDebuggingLayer::SetRepresentationArrows(Representation representation)
{
	if (nullptr != arrowActor)
	{
		SetActorRepresentation(renderer, arrowActor, representation);
	}
}

void VisualDebuggingLayer::ToggleArrowsRepresentation()
{
	if (nullptr != arrowActor)
	{
		ToggleActorRepresentation(renderer, arrowActor);
	}
}

void VisualDebuggingLayer::ShowWiredBoxes(bool show)
{
	if (nullptr != wiredBoxActor)
	{
		ShowActor(renderer, wiredBoxActor, show);
	}
}

void VisualDebuggingLayer::ToggleWiredBoxes()
{
	if (nullptr != wiredBoxActor)
	{
		ToggleActorVisibility(renderer, wiredBoxActor);
	}
}

void VisualDebuggingLayer::SetRepresentationWiredBoxes(Representation representation)
{
	if (nullptr != wiredBoxActor)
	{
		SetActorRepresentation(renderer, wiredBoxActor, representation);
	}
}

void VisualDebuggingLayer::ToggleWiredBoxesRepresentation()
{
	if (nullptr != wiredBoxActor)
	{
		ToggleActorRepresentation(renderer, wiredBoxActor);
	}
}

float VisualDebuggingLayer::GetPointSize()
{
	return pointActor->GetProperty()->GetPointSize();
}

void VisualDebuggingLayer::SetPointSize(float size)
{
	pointActor->GetProperty()->SetPointSize(size);
}

float VisualDebuggingLayer::GetLineWidth()
{
	return pointActor->GetProperty()->GetLineWidth();
}

void VisualDebuggingLayer::SetLineWidth(float width)
{
	lineActor->GetProperty()->SetLineWidth(width);
}

void VisualDebuggingLayer::DrawPoints()
{
	if (lineInfosToDraw.empty())
		return;

	vtkNew<vtkPoints> points;
	vtkNew<vtkUnsignedCharArray> colors;
	colors->SetNumberOfComponents(3);

	for (auto& pointInfo : pointInfosToDraw)
	{
		auto p = std::get<0>(pointInfo);
		auto color = std::get<1>(pointInfo);

		colors->InsertNextTypedTuple(color.data());
		colors->InsertNextTypedTuple(color.data());
	}

	vtkSmartPointer<vtkPolyData> newPointPolyData =
		vtkSmartPointer<vtkPolyData>::New();
	newPointPolyData->SetPoints(points);
	newPointPolyData->GetPointData()->SetScalars(colors);

	vtkSmartPointer<vtkAppendPolyData> appendFilter =
		vtkSmartPointer<vtkAppendPolyData>::New();
	appendFilter->AddInputData(newPointPolyData);
	appendFilter->Update();

	pointPolyData->ShallowCopy(appendFilter->GetOutput());

	pointInfosToDraw.clear();
}

void VisualDebuggingLayer::DrawLines()
{
	if (lineInfosToDraw.empty())
		return;

	vtkNew<vtkPoints> points;
	vtkNew<vtkCellArray> lines;
	vtkNew<vtkUnsignedCharArray> colors;
	colors->SetNumberOfComponents(3);

	for (auto& lineInfo : lineInfosToDraw)
	{
		auto p0 = std::get<0>(lineInfo);
		auto p1 = std::get<1>(lineInfo);
		auto color = std::get<2>(lineInfo);
		
		auto pi0 = points->InsertNextPoint(p0.data());
		auto pi1 = points->InsertNextPoint(p1.data());

		vtkIdType pids[] = { pi0, pi1 };

		lines->InsertNextCell(2, pids);

		colors->InsertNextTypedTuple(color.data());
		colors->InsertNextTypedTuple(color.data());
	}

	vtkSmartPointer<vtkPolyData> newLinePolyData =
		vtkSmartPointer<vtkPolyData>::New();
	newLinePolyData->SetPoints(points);
	newLinePolyData->SetLines(lines);
	newLinePolyData->GetPointData()->SetScalars(colors);

	vtkSmartPointer<vtkAppendPolyData> appendFilter =
		vtkSmartPointer<vtkAppendPolyData>::New();
	appendFilter->AddInputData(linePolyData);
	appendFilter->AddInputData(newLinePolyData);
	appendFilter->Update();

	linePolyData->ShallowCopy(appendFilter->GetOutput());

	lineInfosToDraw.clear();
}

void VisualDebuggingLayer::DrawTriangle()
{
	if (triangleInfosToDraw.empty())
		return;

	vtkNew<vtkPoints> points;
	vtkNew<vtkCellArray> triangles;
	vtkNew<vtkUnsignedCharArray> colors;
	colors->SetNumberOfComponents(3);

	for (auto& triangleInfo : triangleInfosToDraw)
	{
		auto p0 = std::get<0>(triangleInfo);
		auto p1 = std::get<1>(triangleInfo);
		auto p2 = std::get<2>(triangleInfo);
		auto color = std::get<3>(triangleInfo);

		auto pi0 = points->InsertNextPoint(p0.data());
		auto pi1 = points->InsertNextPoint(p1.data());
		auto pi2 = points->InsertNextPoint(p2.data());

		vtkIdType pids[] = { pi0, pi1, pi2 };

		triangles->InsertNextCell(3, pids);

		colors->InsertNextTypedTuple(color.data());
		colors->InsertNextTypedTuple(color.data());
		colors->InsertNextTypedTuple(color.data());
	}

	vtkNew<vtkPolyData> newTrianglePolyData;
	newTrianglePolyData->SetPoints(points);
	newTrianglePolyData->SetPolys(triangles);
	newTrianglePolyData->GetPointData()->SetScalars(colors);

	vtkNew<vtkAppendPolyData> appendFilter;
	appendFilter->AddInputData(trianglePolyData);
	appendFilter->AddInputData(newTrianglePolyData);
	appendFilter->Update();

	trianglePolyData->ShallowCopy(appendFilter->GetOutput());

	triangleInfosToDraw.clear();
}

void VisualDebuggingLayer::DrawSpheres()
{
	if (sphereInfosToDraw.empty())
		return;

	auto points = spherePolyData->GetPoints();
	auto pointData = spherePolyData->GetPointData();
	vtkDoubleArray* scales =
		vtkDoubleArray::SafeDownCast(pointData->GetArray("Scales"));
	vtkDoubleArray* normals =
		vtkDoubleArray::SafeDownCast(pointData->GetArray("Normals"));
	vtkUnsignedCharArray* colors =
		vtkUnsignedCharArray::SafeDownCast(pointData->GetArray("Colors"));

	for (auto& sphereInfo : sphereInfosToDraw)
	{
		auto center = std::get<0>(sphereInfo);
		auto scale = std::get<1>(sphereInfo);
		auto normal = std::get<2>(sphereInfo);
		auto color = std::get<3>(sphereInfo);
		
		points->InsertNextPoint(center.data());
		//scales->InsertNextValue(scale);
		scales->InsertNextTuple3(scale.x() * 2.0f, scale.y() * 2.0f, scale.z() * 2.0f);
		normals->InsertNextTuple3(normal.x(), normal.y(), normal.z());
		colors->InsertNextTypedTuple(color.data());
	}

	points->Modified();
	spherePolyDataMapper->Update();

	sphereInfosToDraw.clear();
}

void VisualDebuggingLayer::DrawCubes()
{
	if (cubeInfosToDraw.empty())
		return;

	auto points = cubePolyData->GetPoints();
	auto pointData = cubePolyData->GetPointData();
	vtkDoubleArray* scales =
		vtkDoubleArray::SafeDownCast(pointData->GetArray("Scales"));
	vtkDoubleArray* normals =
		vtkDoubleArray::SafeDownCast(pointData->GetArray("Normals"));
	vtkUnsignedCharArray* colors =
		vtkUnsignedCharArray::SafeDownCast(pointData->GetArray("Colors"));

	for (auto& cubeInfo : cubeInfosToDraw)
	{
		auto center = std::get<0>(cubeInfo);
		auto scale = std::get<1>(cubeInfo);
		auto normal = std::get<2>(cubeInfo);
		auto color = std::get<3>(cubeInfo);
		
		points->InsertNextPoint(center.data());
		//scales->InsertNextValue(scale);
		scales->InsertNextTuple3(scale.x() * 2.0f, scale.y() * 2.0f, scale.z() * 2.0f);
		normals->InsertNextTuple3(normal.x(), normal.y(), normal.z());
		colors->InsertNextTypedTuple(color.data());
	}

	points->Modified();
	cubePolyDataMapper->Update();

	cubeInfosToDraw.clear();
}

void VisualDebuggingLayer::DrawGlyphs()
{
	if (glyphInfosToDraw.empty())
		return;

	auto points = glyphPolyData->GetPoints();
	auto pointData = glyphPolyData->GetPointData();
	vtkDoubleArray* scales =
		vtkDoubleArray::SafeDownCast(pointData->GetArray("Scales"));
	vtkDoubleArray* normals =
		vtkDoubleArray::SafeDownCast(pointData->GetArray("Normals"));
	vtkUnsignedCharArray* colors =
		vtkUnsignedCharArray::SafeDownCast(pointData->GetArray("Colors"));

	for (auto& glyphInfo : glyphInfosToDraw)
	{
		auto center = std::get<0>(glyphInfo);
		auto scale = std::get<1>(glyphInfo);
		auto normal = std::get<2>(glyphInfo);
		auto color = std::get<3>(glyphInfo);
		
		points->InsertNextPoint(center.data());
		//scales->InsertNextValue(scale);
		scales->InsertNextTuple3(scale.x(), scale.y(), scale.z());
		normals->InsertNextTuple3(normal.x(), normal.y(), normal.z());
		colors->InsertNextTypedTuple(color.data());
	}

	points->Modified();
	glyphPolyDataMapper->Update();

	glyphInfosToDraw.clear();
}

void VisualDebuggingLayer::DrawArrows()
{
	if (arrowInfosToDraw.empty())
		return;

	auto points = arrowPolyData->GetPoints();
	auto pointData = arrowPolyData->GetPointData();
	vtkDoubleArray* scales =vtkDoubleArray::SafeDownCast(pointData->GetArray("Scales"));
	vtkDoubleArray* normals = vtkDoubleArray::SafeDownCast(pointData->GetArray("Normals"));
	vtkUnsignedCharArray* colors = vtkUnsignedCharArray::SafeDownCast(pointData->GetArray("Colors"));

	for (auto& arrowInfo : arrowInfosToDraw)
	{
		auto center = std::get<0>(arrowInfo);
		auto normal = std::get<1>(arrowInfo);
		auto scale = std::get<2>(arrowInfo);
		auto color = std::get<3>(arrowInfo);
		
		points->InsertNextPoint(center.data());
		scales->InsertNextTuple3(scale, scale, scale);
		normals->InsertNextTuple3(normal.x(), normal.y(), normal.z());
		colors->InsertNextTypedTuple(color.data());
	}

	points->Modified();
	arrowGlyph3D->Update();

	arrowInfosToDraw.clear();
}

void VisualDebuggingLayer::DrawWiredBoxes()
{
	if (wiredBoxInfosToDraw.empty())
		return;

	auto points = wiredBoxPolyData->GetPoints();
	auto pointData = wiredBoxPolyData->GetPointData();
	vtkDoubleArray* scales = vtkDoubleArray::SafeDownCast(pointData->GetArray("Scales"));
	vtkDoubleArray* normals = vtkDoubleArray::SafeDownCast(pointData->GetArray("Normals"));
	vtkUnsignedCharArray* colors = vtkUnsignedCharArray::SafeDownCast(pointData->GetArray("Colors"));

	for (auto& wiredBoxInfo : wiredBoxInfosToDraw)
	{
		auto center = std::get<0>(wiredBoxInfo);
		//auto normal = std::get<1>(wiredBoxInfo);
		auto scale = std::get<1>(wiredBoxInfo);
		auto color = std::get<2>(wiredBoxInfo);

		points->InsertNextPoint(center.data());
		scales->InsertNextTuple3(scale.x(), scale.y(), scale.z());
		normals->InsertNextTuple3(0, 0, 1.0f);
		colors->InsertNextTypedTuple(color.data());
	}

	points->Modified();
	scales->Modified();
	normals->Modified();
	colors->Modified();

	wiredBoxGlyph3D->Update();

	wiredBoxInfosToDraw.clear();
	return;

	//if (wiredBoxInfosToDraw.empty())
	//	return;

	//auto points = cubePolyData->GetPoints();
	//auto pointData = cubePolyData->GetPointData();
	//vtkDoubleArray* scales =
	//	vtkDoubleArray::SafeDownCast(pointData->GetArray("Scales"));
	//vtkDoubleArray* normals =
	//	vtkDoubleArray::SafeDownCast(pointData->GetArray("Normals"));
	//vtkUnsignedCharArray* colors =
	//	vtkUnsignedCharArray::SafeDownCast(pointData->GetArray("Colors"));

	//vtkNew<vtkPoints> points;
	//vtkNew<vtkCellArray> lines;
	//vtkNew<vtkUnsignedCharArray> colors;
	//colors->SetNumberOfComponents(3);

	//for (auto& cubeInfo : wiredBoxInfosToDraw)
	//{
	//	auto boxMin = std::get<0>(cubeInfo);
	//	auto boxMax = std::get<1>(cubeInfo);
	//	auto color = std::get<2>(cubeInfo);

	//	auto pi0 = points->InsertNextPoint(boxMin.x(), boxMin.y(), boxMin.z());
	//	auto pi1 = points->InsertNextPoint(boxMax.x(), boxMin.y(), boxMin.z());
	//	auto pi2 = points->InsertNextPoint(boxMin.x(), boxMax.y(), boxMin.z());
	//	auto pi3 = points->InsertNextPoint(boxMax.x(), boxMax.y(), boxMin.z());
	//	auto pi4 = points->InsertNextPoint(boxMin.x(), boxMin.y(), boxMax.z());
	//	auto pi5 = points->InsertNextPoint(boxMax.x(), boxMin.y(), boxMax.z());
	//	auto pi6 = points->InsertNextPoint(boxMin.x(), boxMax.y(), boxMax.z());
	//	auto pi7 = points->InsertNextPoint(boxMax.x(), boxMax.y(), boxMax.z());

	//	vtkIdType pids[] = {
	//		pi0, pi1, pi1, pi3, pi3, pi2, pi2, pi0,
	//		pi4, pi5, pi5, pi7, pi7, pi6, pi6, pi4,
	//		pi0, pi4, pi1, pi5, pi2, pi6, pi3, pi7
	//	};

	//	lines->InsertNextCell(12, pids);

	//	colors->InsertNextTypedTuple(color.data());
	//	colors->InsertNextTypedTuple(color.data());
	//	colors->InsertNextTypedTuple(color.data());
	//	colors->InsertNextTypedTuple(color.data());
	//	colors->InsertNextTypedTuple(color.data());
	//	colors->InsertNextTypedTuple(color.data());
	//	colors->InsertNextTypedTuple(color.data());
	//	colors->InsertNextTypedTuple(color.data());
	//}

	//vtkSmartPointer<vtkPolyData> newWiredBoxLinePolyData =
	//	vtkSmartPointer<vtkPolyData>::New();
	//newWiredBoxLinePolyData->SetPoints(points);
	//newWiredBoxLinePolyData->SetLines(lines);
	//newWiredBoxLinePolyData->GetPointData()->SetScalars(colors);

	//vtkSmartPointer<vtkAppendPolyData> appendFilter =
	//	vtkSmartPointer<vtkAppendPolyData>::New();
	//appendFilter->AddInputData(linePolyData);
	//appendFilter->AddInputData(newWiredBoxLinePolyData);
	//appendFilter->Update();

	//linePolyData->ShallowCopy(appendFilter->GetOutput());

	//lineInfosToDraw.clear();

	//points->Modified();
	//cubePolyDataMapper->Update();

	//cubeInfosToDraw.clear();
}