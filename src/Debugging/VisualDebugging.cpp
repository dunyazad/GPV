#include <Debugging/VisualDebugging.h>
#include <Debugging/VisualDebuggingLayer.h>

bool VisualDebugging::s_needToRender = false;
VisualDebugging* VisualDebugging::s_instance = nullptr;

map<string, int> VisualDebugging::s_layerNameIndexMapping;
vector<VisualDebuggingLayer*> VisualDebugging::s_layers;

vtkSmartPointer<vtkRenderer> VisualDebugging::s_renderer = nullptr;
vtkSmartPointer<vtkRenderWindow> VisualDebugging::s_renderWindow = nullptr;

VisualDebugging::VisualDebugging()
{
}

VisualDebugging::~VisualDebugging()
{
}

void VisualDebugging::Initialize(vtkSmartPointer<vtkRenderer> renderer)
{
	if (nullptr == s_instance)
	{
		s_instance = new VisualDebugging();
	}

	s_renderer = renderer;
	s_renderWindow = s_renderer->GetRenderWindow();
}

void VisualDebugging::Terminate()
{
	for (auto layer : s_layers)
	{
		layer->Terminate();
		delete layer;
	}

	s_layers.clear();
	s_layerNameIndexMapping.clear();
}

VisualDebuggingLayer* VisualDebugging::CreateLayer(const string& layerName)
{
	if (s_layerNameIndexMapping.count(layerName) != 0)
	{
		return s_layers[s_layerNameIndexMapping[layerName]];
	}
	else
	{
		auto layer = new VisualDebuggingLayer(layerName);
		layer->Initialize(s_renderer);
		s_layers.push_back(layer);
		s_layerNameIndexMapping[layerName] = (int)s_layers.size() - 1;
		return layer;
	}
}

void VisualDebugging::Update()
{
	for (auto layer : s_layers)
	{
		layer->Update();
	}

	if (s_needToRender)
	{
		s_renderWindow->Render();
		s_needToRender = false;
	}
}

void VisualDebugging::ClearAll()
{
	for (auto layer : s_layers)
	{
		layer->Clear();
	}

	s_needToRender = true;
}

void VisualDebugging::Clear(const string& layerName)
{
	auto layer = GetLayer(layerName);
	if (nullptr != layer)
	{
		layer->Clear();
	}

	s_needToRender = true;
}

void VisualDebugging::ToggleVisibilityAll()
{
	for (auto layer : s_layers)
	{
		layer->ToggleVisibilityAll();
	}

	s_needToRender = true;
}

void VisualDebugging::SetVisibilityAll(bool visible)
{
	for (auto layer : s_layers)
	{
		layer->ShowAll(visible);
	}

	s_needToRender = true;
}

void VisualDebugging::ToggleVisibility(const string& layerName)
{
	auto layer = GetLayer(layerName);
	if (nullptr != layer)
	{
		layer->ToggleVisibilityAll();
	}

	s_needToRender = true;
}

void VisualDebugging::SetVisibility(const string& layerName, bool visible)
{
	auto layer = GetLayer(layerName);
	if (nullptr != layer)
	{
		layer->ShowAll(visible);
	}

	s_needToRender = true;
}

void VisualDebugging::ToggleVisibilityByIndex(int index)
{
	if (index < s_layers.size())
	{
		s_layers[index]->ToggleVisibilityAll();
	}

	s_needToRender = true;
}

void VisualDebugging::SetVisibilityByIndex(int index, bool visible)
{
	if (index < s_layers.size())
	{
		s_layers[index]->ShowAll(visible);
	}

	s_needToRender = true;
}

void VisualDebugging::SetRepresentationAll(Representation representation)
{
	for (auto layer : s_layers)
	{
		layer->SetRepresentationAll((VisualDebuggingLayer::Representation)representation);
	}

	s_needToRender = true;
}

void VisualDebugging::SetRepresentation(const string& layerName, Representation representation)
{
	auto layer = GetLayer(layerName);
	if (nullptr != layer)
	{
		layer->SetRepresentationAll((VisualDebuggingLayer::Representation)representation);
	}

	s_needToRender = true;
}

void VisualDebugging::SetRepresentationByIndex(int index, Representation representation)
{
	if (index < s_layers.size())
	{
		s_layers[index]->SetRepresentationAll((VisualDebuggingLayer::Representation)representation);
	}

	s_needToRender = true;
}

void VisualDebugging::ToggleRepresentationAll()
{
	for (auto layer : s_layers)
	{
		layer->ToggleAllRepresentation();
	}

	s_needToRender = true;
}

void VisualDebugging::ToggleRepresentation(const string& layerName)
{
	auto layer = GetLayer(layerName);
	if (nullptr != layer)
	{
		layer->ToggleAllRepresentation();
	}

	s_needToRender = true;
}

void VisualDebugging::ToggleRepresentationByIndex(int index)
{
	if (index < s_layers.size())
	{
		s_layers[index]->ToggleAllRepresentation();
	}

	s_needToRender = true;
}

float VisualDebugging::GetPointSize(const string& layerName)
{
	auto layer = GetLayer(layerName);
	if (nullptr != layer)
	{
		return layer->GetPointSize();
	}
	else
	{
		return 0.0f;
	}
}

void VisualDebugging::SetPointSize(const string& layerName, float size)
{
	auto layer = GetLayer(layerName);
	if (nullptr != layer)
	{
		layer->SetPointSize(size);
	}

	s_needToRender = true;
}

float VisualDebugging::GetLineWidth(const string& layerName)
{
	auto layer = GetLayer(layerName);
	if (nullptr != layer)
	{
		return layer->GetLineWidth();
	}
	else
	{
		return 0.0f;
	}
}

void VisualDebugging::SetLineWidth(const string& layerName, float width)
{
	auto layer = GetLayer(layerName);
	if (nullptr != layer)
	{
		layer->SetLineWidth(width);
	}

	s_needToRender = true;
}

vtkSmartPointer<vtkActor> VisualDebugging::GetPointActor(const string& layerName)
{
	auto layer = GetLayer(layerName);
	if (nullptr != layer)
	{
		return layer->GetPointActor();
	}
	else
	{
		return nullptr;
	}
}

vtkSmartPointer<vtkActor> VisualDebugging::GetLineActor(const string& layerName)
{
	auto layer = GetLayer(layerName);
	if (nullptr != layer)
	{
		return layer->GetLineActor();
	}
	else
	{
		return nullptr;
	}
}

vtkSmartPointer<vtkActor> VisualDebugging::GetTriangleActor(const string& layerName)
{
	auto layer = GetLayer(layerName);
	if (nullptr != layer)
	{
		return layer->GetTriangleActor();
	}
	else
	{
		return nullptr;
	}
}

vtkSmartPointer<vtkActor> VisualDebugging::GetSphereActor(const string& layerName)
{
	auto layer = GetLayer(layerName);
	if (nullptr != layer)
	{
		return layer->GetSphereActor();
	}
	else
	{
		return nullptr;
	}
}

vtkSmartPointer<vtkActor> VisualDebugging::GetCubeActor(const string& layerName)
{
	auto layer = GetLayer(layerName);
	if (nullptr != layer)
	{
		return layer->GetCubeActor();
	}
	else
	{
		return nullptr;
	}
}

vtkSmartPointer<vtkActor> VisualDebugging::GetGlyphActor(const string& layerName)
{
	auto layer = GetLayer(layerName);
	if (nullptr != layer)
	{
		return layer->GetGlyphActor();
	}
	else
	{
		return nullptr;
	}
}

vtkSmartPointer<vtkActor> VisualDebugging::GetArrowActor(const string& layerName)
{
	auto layer = GetLayer(layerName);
	if (nullptr != layer)
	{
		return layer->GetArrowActor();
	}
	else
	{
		return nullptr;
	}
}

vtkSmartPointer<vtkActor> VisualDebugging::GetWiredBoxActor(const string& layerName)
{
	auto layer = GetLayer(layerName);
	if (nullptr != layer)
	{
		return layer->GetWiredBoxActor();
	}
	else
	{
		return nullptr;
	}
}

int VisualDebugging::GetNumberOfLayers()
{
	return (int)s_layers.size();
}

VisualDebuggingLayer* VisualDebugging::GetLayer(const string& layerName)
{
	if (s_layerNameIndexMapping.count(layerName) != 0)
	{
		return s_layers[s_layerNameIndexMapping[layerName]];
	}
	else
	{
		return nullptr;
	}
}

void VisualDebugging::AddLine(const string& layerName, const Eigen::Vector3f& p0, const Eigen::Vector3f& p1, const Color4& color)
{
	auto layer = GetLayer(layerName);
	if (nullptr == layer)
	{
		layer = CreateLayer(layerName);
	}

	layer->AddLine(p0, p1, color);

	s_needToRender = true;
}

void VisualDebugging::AddTriangle(const string& layerName, const Eigen::Vector3f& p0, const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, const Color4& color)
{
	auto layer = GetLayer(layerName);
	if (nullptr == layer)
	{
		layer = CreateLayer(layerName);
	}

	layer->AddTriangle(p0, p1, p2, color);

	s_needToRender = true;
}

void VisualDebugging::AddSphere(const string& layerName, const Eigen::Vector3f& center, const Eigen::Vector3f& scale, const Eigen::Vector3f& normal, const Color4& color)
{
	auto layer = GetLayer(layerName);
	if (nullptr == layer)
	{
		layer = CreateLayer(layerName);
	}

	layer->AddSphere(center, scale, normal, color);

	s_needToRender = true;
}

void VisualDebugging::AddCube(const string& layerName, const Eigen::Vector3f& center, const Color4& color)
{
	AddCube(layerName, center, { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 0.0f }, color);
}

void VisualDebugging::AddCube(const string& layerName, const Eigen::Vector3f& center, float scale, const Color4& color)
{
	AddCube(layerName, center, { scale, scale, scale }, { 0.0f, 0.0f, 0.0f }, color);
}

void VisualDebugging::AddCube(const string& layerName, const Eigen::Vector3f& center, const Eigen::Vector3f& scale, const Eigen::Vector3f& normal, const Color4& color)
{
	auto layer = GetLayer(layerName);
	if (nullptr == layer)
	{
		layer = CreateLayer(layerName);
	}

	layer->AddCube(center, scale, normal, color);

	s_needToRender = true;
}

void VisualDebugging::AddGlyph(const string& layerName, const Eigen::Vector3f& center, const Eigen::Vector3f& scale, const Eigen::Vector3f& normal, const Color4& color)
{
	auto layer = GetLayer(layerName);
	if (nullptr == layer)
	{
		layer = CreateLayer(layerName);
	}

	layer->AddGlyph(center, scale, normal, color);

	s_needToRender = true;
}

void VisualDebugging::AddArrow(const string& layerName, const Eigen::Vector3f& center, const Eigen::Vector3f& normal, float scale, const Color4& color)
{
	auto layer = GetLayer(layerName);
	if (nullptr == layer)
	{
		layer = CreateLayer(layerName);
	}

	layer->AddArrow(center, normal, scale, color);

	s_needToRender = true;
}

void VisualDebugging::AddWiredBox(const string& layerName, const Eigen::Vector3f& boxMin, const Eigen::Vector3f& boxMax, const Color4& color)
{
	auto layer = GetLayer(layerName);
	if (nullptr == layer)
	{
		layer = CreateLayer(layerName);
	}

	layer->AddWiredBox(boxMin, boxMax, color);

	s_needToRender = true;
}
