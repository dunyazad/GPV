#pragma once

#include <Common.h>
#include <Color.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

class VisualDebugging;

class VisualDebuggingLayerElement
{
public:
	VisualDebuggingLayerElement(vtkSmartPointer<vtkRenderer> renderer);
	virtual ~VisualDebuggingLayerElement();

	vtkSmartPointer<vtkPolyData> polyData;
	vtkSmartPointer<vtkPolyDataMapper> polyDataMapper;
	vtkSmartPointer<vtkActor> actor;

	vtkSmartPointer<vtkRenderer> renderer;
};

class VisualDebuggingLayerElementGlyph : public VisualDebuggingLayerElement
{
public:
	VisualDebuggingLayerElementGlyph(vtkSmartPointer<vtkRenderer> renderer);
	virtual ~VisualDebuggingLayerElementGlyph();

	vtkSmartPointer<vtkGlyph3D> glyph;
	vtkSmartPointer<vtkGlyph3DMapper> glyphMapper;
};

class VisualDebuggingLayer
{
public:
	enum Representation { HPoints, HWireFrame, HSurface };

	VisualDebuggingLayer(const string& layerName);
	~VisualDebuggingLayer();

	void Initialize(vtkSmartPointer<vtkRenderer> renderer);
	void Terminate();

	void AddPoint(const Eigen::Vector3f& p, const Color4& color);

	void AddLine(const Eigen::Vector3f& p0, const Eigen::Vector3f& p1, const Color4& color);
	
	void AddTriangle(const Eigen::Vector3f& p0, const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, const Color4& color);

	void AddSphere(const Eigen::Vector3f& center, const Eigen::Vector3f& scale, const Eigen::Vector3f& normal, const Color4& color);
	
	void AddCube(const Eigen::Vector3f& center, const Eigen::Vector3f& scale, const Eigen::Vector3f& normal, const Color4& color);

	void AddGlyph(const Eigen::Vector3f& center, const Eigen::Vector3f& scale, const Eigen::Vector3f& normal, const Color4& color);

	void AddArrow(const Eigen::Vector3f& center, const Eigen::Vector3f& normal, float scale, const Color4& color);

	void AddGrid(const Eigen::Vector3f& position, const Eigen::Vector3f& normal, float width, float height, float interval, const Color4& color);

	void Update();
	void Clear();

	void ShowAll(bool show);
	void ToggleVisibilityAll();
	void SetRepresentationAll(Representation representation);
	void ToggleAllRepresentation();

	float GetPointSize();
	void SetPointSize(float size);
	float GetLineWidth();
	void SetLineWidth(float width);

protected:
	inline void ShowActor(vtkSmartPointer<vtkRenderer> renderer, vtkSmartPointer<vtkActor> actor, bool show)
	{
		if (nullptr != actor)
		{
			actor->SetVisibility(show);
			//renderer->GetRenderWindow()->Render();
		}
	}

	inline void ToggleActorVisibility(vtkSmartPointer<vtkRenderer> renderer, vtkSmartPointer<vtkActor> actor)
	{
		if (nullptr != actor)
		{
			actor->SetVisibility(!actor->GetVisibility());
			//renderer->GetRenderWindow()->Render();
		}
	}

	inline void SetActorRepresentation(vtkSmartPointer<vtkRenderer> renderer, vtkSmartPointer<vtkActor> actor, Representation representation)
	{
		if (nullptr != actor)
		{
			actor->GetProperty()->SetRepresentation(representation);
			//renderer->GetRenderWindow()->Render();
		}
	}

	inline void ToggleActorRepresentation(vtkSmartPointer<vtkRenderer> renderer, vtkSmartPointer<vtkActor> actor)
	{
		if (nullptr != actor)
		{
			auto mode = actor->GetProperty()->GetRepresentation();
			mode += 1;
			if (mode > VTK_SURFACE)
			{
				mode = VTK_POINTS;
			}
			actor->GetProperty()->SetRepresentation(mode);
			//renderer->GetRenderWindow()->Render();
		}
	}

private:
	string layerName = "";
	vtkSmartPointer<vtkRenderer> renderer;
	vtkSmartPointer<vtkRenderWindow> renderWindow;

	map<string, VisualDebuggingLayerElement*> elements;

	//vtkSmartPointer<vtkActor> pointActor;
	//vtkSmartPointer<vtkPolyDataMapper> pointPolyDataMapper;
	//vtkSmartPointer<vtkPolyData> pointPolyData;

	//vtkSmartPointer<vtkActor> lineActor;
	//vtkSmartPointer<vtkPolyDataMapper> linePolyDataMapper;
	//vtkSmartPointer<vtkPolyData> linePolyData;

	//vtkSmartPointer<vtkActor> triangleActor;
	//vtkSmartPointer<vtkPolyDataMapper> trianglePolyDataMapper;
	//vtkSmartPointer<vtkPolyData> trianglePolyData;

	//vtkSmartPointer<vtkActor> sphereActor;
	//vtkSmartPointer<vtkGlyph3DMapper> spherePolyDataMapper;
	//vtkSmartPointer<vtkPolyData> spherePolyData;

	//vtkSmartPointer<vtkActor> cubeActor;
	//vtkSmartPointer<vtkGlyph3DMapper> cubePolyDataMapper;
	//vtkSmartPointer<vtkPolyData> cubePolyData;

	//vtkSmartPointer<vtkActor> glyphActor;
	//vtkSmartPointer<vtkGlyph3DMapper> glyphPolyDataMapper;
	//vtkSmartPointer<vtkPolyData> glyphPolyData;

	//vtkSmartPointer<vtkActor> arrowActor;
	//vtkSmartPointer<vtkPolyDataMapper> arrowPolyDataMapper;
	//vtkSmartPointer<vtkGlyph3D> arrowGlyph3D;
	//vtkSmartPointer<vtkPolyData> arrowPolyData;

	//vtkSmartPointer<vtkActor> wiredBoxActor;
	//vtkSmartPointer<vtkPolyDataMapper> wiredBoxPolyDataMapper;
	//vtkSmartPointer<vtkGlyph3D> wiredBoxGlyph3D;
	//vtkSmartPointer<vtkPolyData> wiredBoxPolyData;

	//vtkSmartPointer<vtkActor> gridActor;
	//vtkSmartPointer<vtkPolyDataMapper> gridPolyDataMapper;
	//vtkSmartPointer<vtkPolyData> gridPolyData;

	void DrawPoints();
	void DrawLines();
	void DrawTriangle();
	void DrawSpheres();
	void DrawCubes();
	void DrawGlyphs();
	void DrawArrows();
	void DrawGrids();

	vector<std::tuple<Eigen::Vector3f, Color4>> pointInfosToDraw;
	vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, Color4>> lineInfosToDraw;
	vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f, Color4>> triangleInfosToDraw;
	vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f, Color4>> sphereInfosToDraw;
	vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f, Color4>> cubeInfosToDraw;
	vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f, Color4>> glyphInfosToDraw;
	vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, float, Color4>> arrowInfosToDraw;
	vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, float, float, float, Color4>> gridInfosToDraw;

public:
	friend class VisualDebugging;
};
