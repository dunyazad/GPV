#include <Debugging/VisualDebugging.h>

bool VisualDebugging::s_needToRender = false;
VisualDebugging *VisualDebugging::s_instance = nullptr;

map<string, int> VisualDebugging::s_layerNameIndexMapping;
vector<VisualDebuggingLayer *> VisualDebugging::s_layers;

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

VisualDebuggingLayer* VisualDebugging::CreateLayer(const string &layerName)
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

void VisualDebugging::Clear(const string &layerName)
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

void VisualDebugging::ToggleVisibility(const string &layerName)
{
	auto layer = GetLayer(layerName);
	if (nullptr != layer)
	{
		layer->ToggleVisibilityAll();
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

void VisualDebugging::SetRepresentationAll(Representation representation)
{
	for (auto layer : s_layers)
	{
		layer->SetRepresentationAll(representation);
	}

	s_needToRender = true;
}

void VisualDebugging::SetRepresentation(const string &layerName, Representation representation)
{
	auto layer = GetLayer(layerName);
	if (nullptr != layer)
	{
		layer->SetRepresentationAll(representation);
	}

	s_needToRender = true;
}

void VisualDebugging::SetRepresentationByIndex(int index, Representation representation)
{
	if (index < s_layers.size())
	{
		s_layers[index]->SetRepresentationAll(representation);
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

void VisualDebugging::ToggleRepresentation(const string &layerName)
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

int VisualDebugging::GetNumberOfLayers()
{
	return (int)s_layers.size();
}

VisualDebuggingLayer *VisualDebugging::GetLayer(const string &layerName)
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

void VisualDebugging::AddLine(const string &layerName, const Eigen::Vector3f &p0, const Eigen::Vector3f &p1, unsigned char r, unsigned char g, unsigned char b)
{
	auto layer = GetLayer(layerName);
	if (nullptr == layer)
	{
		layer = CreateLayer(layerName);
	}

	layer->AddLine(p0, p1, r, g, b);

	s_needToRender = true;
}

void VisualDebugging::AddTriangle(const string &layerName, const Eigen::Vector3f &p0, const Eigen::Vector3f &p1, const Eigen::Vector3f &p2, unsigned char r, unsigned char g, unsigned char b)
{
	auto layer = GetLayer(layerName);
	if (nullptr == layer)
	{
		layer = CreateLayer(layerName);
	}

	layer->AddTriangle(p0, p1, p2, r, g, b);

	s_needToRender = true;
}

void VisualDebugging::AddSphere(const string& layerName, const Eigen::Vector3f& center, const Eigen::Vector3f& scale, const Eigen::Vector3f& normal, unsigned char r, unsigned char g, unsigned char b)
{
	auto layer = GetLayer(layerName);
	if (nullptr == layer)
	{
		layer = CreateLayer(layerName);
	}

	layer->AddSphere(center, scale, normal, r, g, b);

	s_needToRender = true;
}

void VisualDebugging::AddCube(const string &layerName, const Eigen::Vector3f &center, const Eigen::Vector3f& scale, const Eigen::Vector3f& normal, unsigned char r, unsigned char g, unsigned char b)
{
	auto layer = GetLayer(layerName);
	if (nullptr == layer)
	{
		layer = CreateLayer(layerName);
	}

	layer->AddCube(center, scale, normal, r, g, b);

	s_needToRender = true;
}

void VisualDebugging::AddGlyph(const string& layerName, const Eigen::Vector3f& center, const Eigen::Vector3f& scale, const Eigen::Vector3f& normal, unsigned char r, unsigned char g, unsigned char b)
{
	auto layer = GetLayer(layerName);
	if (nullptr == layer)
	{
		layer = CreateLayer(layerName);
	}

	layer->AddGlyph(center, scale, normal, r, g, b);

	s_needToRender = true;
}

void VisualDebugging::AddArrow(const string& layerName, const Eigen::Vector3f& center, const Eigen::Vector3f& normal, float scale, unsigned char r, unsigned char g, unsigned char b)
{
	auto layer = GetLayer(layerName);
	if (nullptr == layer)
	{
		layer = CreateLayer(layerName);
	}

	layer->AddArrow(center, normal, scale, r, g, b);

	s_needToRender = true;
}
