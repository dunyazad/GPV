#include <Debugging/VisualDebuggingLayer.h>
#include <Debugging/vtkPolygonalFrustumSource.h>

#include <Algorithm/CustomPolyDataFilter.h>

#include <App/Utility.h>

VisualDebuggingLayerElement::VisualDebuggingLayerElement(vtkSmartPointer<vtkRenderer> renderer)
	: renderer(renderer)
{
	polyData = vtkSmartPointer<vtkPolyData>::New();
	polyDataMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	polyDataMapper->SetInputData(polyData);
	polyDataMapper->SetScalarModeToUsePointData();
	actor = vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(polyDataMapper);
	renderer->AddActor(actor);
}

VisualDebuggingLayerElement::~VisualDebuggingLayerElement()
{
	if (nullptr != actor)
	{
		renderer->RemoveActor(actor);
	}

	polyData = nullptr;
	polyDataMapper = nullptr;
	actor = nullptr;
}

VisualDebuggingLayerElementGlyph::VisualDebuggingLayerElementGlyph(vtkSmartPointer<vtkRenderer> renderer)
	: VisualDebuggingLayerElement(renderer)
{
	glyphMapper = vtkSmartPointer<vtkGlyph3DMapper>::New();
	actor->SetMapper(glyphMapper);
}

VisualDebuggingLayerElementGlyph::~VisualDebuggingLayerElementGlyph()
{
	glyph = nullptr;
	glyphMapper = nullptr;
}

VisualDebuggingLayer::VisualDebuggingLayer(const string& layerName)
	: layerName(layerName) {}

VisualDebuggingLayer::~VisualDebuggingLayer() {}

void VisualDebuggingLayer::Initialize(vtkSmartPointer<vtkRenderer> renderer)
{
	this->renderer = renderer;
	renderWindow = renderer->GetRenderWindow();

#pragma region Point
	{
		auto element = new VisualDebuggingLayerElement(renderer);
		elements["Points"] = element;

		vtkNew<vtkPoints> points;
		element->polyData->SetPoints(points);

		vtkNew<vtkUnsignedCharArray> colors;
		colors->SetNumberOfComponents(3);
		element->polyData->GetCellData()->SetScalars(colors);
	}
#pragma endregion

#pragma region Line
	{
		auto element = new VisualDebuggingLayerElement(renderer);
		elements["Lines"] = element;

		vtkNew<vtkPoints> points;
		element->polyData->SetPoints(points);

		vtkNew<vtkCellArray> lines;
		element->polyData->SetLines(lines);

		vtkNew<vtkUnsignedCharArray> colors;
		colors->SetNumberOfComponents(3);
		element->polyData->GetCellData()->SetScalars(colors);
	}
#pragma endregion

#pragma region Triangle
	{
		auto element = new VisualDebuggingLayerElement(renderer);
		elements["Triangles"] = element;

		vtkNew<vtkPoints> points;
		element->polyData->SetPoints(points);

		vtkNew<vtkCellArray> triangles;
		element->polyData->SetPolys(triangles);

		vtkNew<vtkUnsignedCharArray> colors;
		colors->SetNumberOfComponents(3);
		element->polyData->GetCellData()->SetScalars(colors);
	}
#pragma endregion

#pragma region Sphere
	{
		auto element = new VisualDebuggingLayerElementGlyph(renderer);
		elements["Spheres"] = element;

		vtkNew<vtkPoints> points;
		element->polyData->SetPoints(points);

		vtkNew<vtkUnsignedCharArray> colors;
		colors->SetName("Colors");
		colors->SetNumberOfComponents(3);
		element->polyData->GetPointData()->AddArray(colors);

		vtkNew<vtkDoubleArray> scales;
		scales->SetName("Scales");
		scales->SetNumberOfComponents(3);
		element->polyData->GetPointData()->AddArray(scales);

		vtkNew<vtkDoubleArray> normals;
		normals->SetNumberOfComponents(3);
		normals->SetName("Normals");
		element->polyData->GetPointData()->AddArray(normals);

		vtkNew<vtkSphereSource> sphereSource;
		sphereSource->Update();

		element->glyphMapper->SetSourceConnection(sphereSource->GetOutputPort());
		element->glyphMapper->SetInputData(element->polyData);
		element->glyphMapper->SetScalarModeToUsePointFieldData();
		element->glyphMapper->SetScaleModeToScaleByVectorComponents();
		element->glyphMapper->SetScaleArray("Scales");
		element->glyphMapper->SelectColorArray("Colors");
		element->glyphMapper->SetOrientationArray("Normals");
		element->glyphMapper->OrientOn();
		element->glyphMapper->Update();
	}
#pragma endregion

#pragma region Cube
	{
		auto element = new VisualDebuggingLayerElementGlyph(renderer);
		elements["Cubes"] = element;

		vtkNew<vtkPoints> points;
		element->polyData->SetPoints(points);

		vtkNew<vtkUnsignedCharArray> colors;
		colors->SetName("Colors");
		colors->SetNumberOfComponents(3);
		element->polyData->GetPointData()->AddArray(colors);

		vtkNew<vtkDoubleArray> scales;
		scales->SetNumberOfComponents(1);
		scales->SetName("Scales");
		scales->SetNumberOfComponents(3);
		element->polyData->GetPointData()->AddArray(scales);

		vtkNew<vtkDoubleArray> normals;
		normals->SetNumberOfComponents(3);
		normals->SetName("Normals");
		element->polyData->GetPointData()->AddArray(normals);

		vtkNew<vtkCubeSource> cubeSource;
		cubeSource->Update();

		element->glyphMapper->SetSourceConnection(cubeSource->GetOutputPort());
		element->glyphMapper->SetInputData(element->polyData);
		element->glyphMapper->SetScalarModeToUsePointFieldData();
		element->glyphMapper->SetScaleModeToScaleByVectorComponents();
		element->glyphMapper->SetScaleArray("Scales");
		element->glyphMapper->SelectColorArray("Colors");
		element->glyphMapper->SetOrientationArray("Normals");
		element->glyphMapper->OrientOn();
		element->glyphMapper->Update();
	}
#pragma endregion

#pragma region Glyph
	{
		auto element = new VisualDebuggingLayerElementGlyph(renderer);
		elements["Glyphs"] = element;

		vtkNew<vtkPoints> points;
		element->polyData->SetPoints(points);

		vtkNew<vtkUnsignedCharArray> colors;
		colors->SetName("Colors");
		colors->SetNumberOfComponents(3);
		element->polyData->GetPointData()->AddArray(colors);

		vtkNew<vtkDoubleArray> scales;
		scales->SetNumberOfComponents(1);
		scales->SetName("Scales");
		scales->SetNumberOfComponents(3);
		element->polyData->GetPointData()->AddArray(scales);

		vtkNew<vtkDoubleArray> normals;
		normals->SetNumberOfComponents(3);
		normals->SetName("Normals");
		element->polyData->GetPointData()->AddArray(normals);

		vtkNew<vtkPolygonalFrustumSource> glyphSource;
		glyphSource->SetNumberOfSides(32);
		glyphSource->SetTopRadius(1.0);
		glyphSource->SetBottomRadius(0.5);
		glyphSource->SetHeight(4.0);
		glyphSource->Update();

		element->glyphMapper->SetSourceConnection(glyphSource->GetOutputPort());
		element->glyphMapper->SetInputData(element->polyData);
		element->glyphMapper->SetScalarModeToUsePointFieldData();
		element->glyphMapper->SetScaleModeToScaleByVectorComponents();
		element->glyphMapper->SetScaleArray("Scales");
		element->glyphMapper->SelectColorArray("Colors");
		element->glyphMapper->SetOrientationArray("Normals");
		element->glyphMapper->OrientOn();
		element->glyphMapper->Update();
	}
#pragma endregion

#pragma region Arrow
	{
		auto element = new VisualDebuggingLayerElementGlyph(renderer);
		elements["Arrows"] = element;

		vtkNew<vtkPoints> points;
		element->polyData->SetPoints(points);

		vtkNew<vtkDoubleArray> scales;
		scales->SetNumberOfComponents(3);
		scales->SetName("Scales");
		element->polyData->GetPointData()->AddArray(scales);

		vtkNew<vtkDoubleArray> normals;
		normals->SetNumberOfComponents(3);
		normals->SetName("Normals");
		element->polyData->GetPointData()->AddArray(normals);

		vtkNew<vtkUnsignedCharArray> colors;
		colors->SetName("Colors");
		colors->SetNumberOfComponents(3);
		element->polyData->GetPointData()->AddArray(colors);

		vtkNew<vtkArrowSource> arrowSource;
		arrowSource->Update();

		element->glyph = vtkSmartPointer<vtkGlyph3D>::New();
		element->glyph->SetSourceConnection(arrowSource->GetOutputPort());
		element->glyph->SetInputData(element->polyData);
		element->glyph->SetScaleModeToScaleByScalar();
		element->glyph->SetColorModeToColorByScalar();

		element->glyph->SetInputArrayToProcess(
			0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS, "Scales");
		element->glyph->SetInputArrayToProcess(
			1, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS, "Normals");
		element->glyph->SetInputArrayToProcess(
			3, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS, "Colors");
		element->glyph->Update();

		element->polyDataMapper->SetInputConnection(element->glyph->GetOutputPort());
	}
#pragma endregion

#pragma region Grid
	{
		auto element = new VisualDebuggingLayerElement(renderer);
		elements["Grids"] = element;

		vtkNew<vtkPoints> points;
		element->polyData->SetPoints(points);

		vtkNew<vtkCellArray> lines;
		element->polyData->SetLines(lines);

		vtkNew<vtkUnsignedCharArray> colors;
		colors->SetNumberOfComponents(3);
		element->polyData->GetCellData()->SetScalars(colors);
	}
#pragma endregion
}

void VisualDebuggingLayer::Terminate()
{
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
	DrawGrids();

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

void VisualDebuggingLayer::AddGrid(const Eigen::Vector3f& position, const Eigen::Vector3f& normal, float width, float height, float interval, const Color4& color)
{
	gridInfosToDraw.push_back(std::make_tuple(position, normal, width, height, interval, color));
}

void VisualDebuggingLayer::ShowAll(bool show)
{
	for (auto& kvp : elements)
	{
		ShowActor(renderer, kvp.second->actor, show);
	}
}

void VisualDebuggingLayer::ToggleVisibilityAll()
{
	for (auto& kvp : elements)
	{
		ToggleActorVisibility(renderer, kvp.second->actor);
	}
}

void VisualDebuggingLayer::SetRepresentationAll(Representation representation)
{
	for (auto& kvp : elements)
	{
		SetActorRepresentation(renderer, kvp.second->actor, representation);
	}
}

void VisualDebuggingLayer::ToggleAllRepresentation()
{
	for (auto& kvp : elements)
	{
		ToggleActorRepresentation(renderer, kvp.second->actor);
	}
}

float VisualDebuggingLayer::GetPointSize()
{
	return elements["Points"]->actor->GetProperty()->GetPointSize();
}

void VisualDebuggingLayer::SetPointSize(float size)
{
	elements["Points"]->actor->GetProperty()->SetPointSize(size);
}

float VisualDebuggingLayer::GetLineWidth()
{
	return elements["Lines"]->actor->GetProperty()->GetLineWidth();
}

void VisualDebuggingLayer::SetLineWidth(float width)
{
	elements["Lines"]->actor->GetProperty()->SetLineWidth(width);
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

	elements["Points"]->polyData->ShallowCopy(appendFilter->GetOutput());

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
	appendFilter->AddInputData(elements["Lines"]->polyData);
	appendFilter->AddInputData(newLinePolyData);
	appendFilter->Update();

	elements["Lines"]->polyData->ShallowCopy(appendFilter->GetOutput());

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
	appendFilter->AddInputData(elements["Triangles"]->polyData);
	appendFilter->AddInputData(newTrianglePolyData);
	appendFilter->Update();

	elements["Triangles"]->polyData->ShallowCopy(appendFilter->GetOutput());

	triangleInfosToDraw.clear();
}

void VisualDebuggingLayer::DrawSpheres()
{
	if (sphereInfosToDraw.empty())
		return;

	auto points = elements["Spheres"]->polyData->GetPoints();
	auto pointData = elements["Spheres"]->polyData->GetPointData();
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
	elements["Lines"]->polyDataMapper->Update();

	sphereInfosToDraw.clear();
}

void VisualDebuggingLayer::DrawCubes()
{
	if (cubeInfosToDraw.empty())
		return;

	auto points = elements["Cubes"]->polyData->GetPoints();
	auto pointData = elements["Cubes"]->polyData->GetPointData();
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
	elements["Cubes"]->polyDataMapper->Update();

	cubeInfosToDraw.clear();
}

void VisualDebuggingLayer::DrawGlyphs()
{
	if (glyphInfosToDraw.empty())
		return;

	auto points = elements["Glyphs"]->polyData->GetPoints();
	auto pointData = elements["Glyphs"]->polyData->GetPointData();
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
	elements["Glyphs"]->polyDataMapper->Update();

	glyphInfosToDraw.clear();
}

void VisualDebuggingLayer::DrawArrows()
{
	if (arrowInfosToDraw.empty())
		return;

	auto points = elements["Arrows"]->polyData->GetPoints();
	auto pointData = elements["Arrows"]->polyData->GetPointData();
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
	elements["Arrows"]->polyDataMapper->Update();

	arrowInfosToDraw.clear();
}

void VisualDebuggingLayer::DrawGrids()
{
	if (gridInfosToDraw.empty())
		return;

	vtkNew<vtkPoints> points;
	vtkNew<vtkCellArray> grids;
	vtkNew<vtkUnsignedCharArray> colors;
	colors->SetNumberOfComponents(3);

	for (auto& gridInfo : gridInfosToDraw)
	{
		auto p = std::get<0>(gridInfo);
		auto n = std::get<1>(gridInfo);
		auto width = std::get<2>(gridInfo);
		auto height = std::get<3>(gridInfo);
		auto interval = std::get<4>(gridInfo);
		auto color = std::get<5>(gridInfo);

		float xmin = -width * 0.5f;
		float ymin = -height * 0.5f;
		float zmin = 0.0f;

		float xmax = width * 0.5f;
		float ymax = height * 0.5f;
		float zmax = 0.0f;

		int xCount = (int)ceilf(width / interval);
		int yCount = (int)ceilf(height / interval);

		auto rotationMatrix = computeRotationMatrix({ 0.0f, 0.0f, 1.0f }, n);

		for (size_t i = 0; i < yCount; i++)
		{
			auto p0 = Eigen::Vector3f(xmin, ymin + (float)i * interval, 0.0f);
			auto p1 = Eigen::Vector3f(xmax, ymin + (float)i * interval, 0.0f);

			Eigen::Vector3f tp0 = rotationMatrix * p0 + p;
			Eigen::Vector3f tp1 = rotationMatrix * p1 + p;

			auto pi0 = points->InsertNextPoint(tp0.data());
			auto pi1 = points->InsertNextPoint(tp1.data());

			vtkIdType pids[] = { pi0, pi1 };

			grids->InsertNextCell(2, pids);

			colors->InsertNextTypedTuple(color.data());
			colors->InsertNextTypedTuple(color.data());
		}

		for (size_t i = 0; i < xCount; i++)
		{
			auto p0 = Eigen::Vector3f(xmin + (float)i * interval, ymin, 0.0f);
			auto p1 = Eigen::Vector3f(xmin + (float)i * interval, ymax, 0.0f);

			Eigen::Vector3f tp0 = rotationMatrix * p0 + p;
			Eigen::Vector3f tp1 = rotationMatrix * p1 + p;

			auto pi0 = points->InsertNextPoint(tp0.data());
			auto pi1 = points->InsertNextPoint(tp1.data());

			vtkIdType pids[] = { pi0, pi1 };

			grids->InsertNextCell(2, pids);

			colors->InsertNextTypedTuple(color.data());
			colors->InsertNextTypedTuple(color.data());
		}
	}

	vtkSmartPointer<vtkPolyData> newgridPolyData =
		vtkSmartPointer<vtkPolyData>::New();
	newgridPolyData->SetPoints(points);
	newgridPolyData->SetLines(grids);
	newgridPolyData->GetPointData()->SetScalars(colors);

	vtkSmartPointer<vtkAppendPolyData> appendFilter =
		vtkSmartPointer<vtkAppendPolyData>::New();
	appendFilter->AddInputData(elements["Grids"]->polyData);
	appendFilter->AddInputData(newgridPolyData);
	appendFilter->Update();

	elements["Grids"]->polyData->ShallowCopy(appendFilter->GetOutput());

	gridInfosToDraw.clear();
}