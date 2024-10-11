#pragma once

#include <Common.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

enum Representation { HPoints, HWireFrame, HSurface };

inline void ShowActor(vtkSmartPointer<vtkRenderer> renderer, vtkSmartPointer<vtkActor> actor, bool show)
{
	if (nullptr != actor)
	{
		actor->SetVisibility(show);
		renderer->GetRenderWindow()->Render();
	}
}

inline void ToggleActorVisibility(vtkSmartPointer<vtkRenderer> renderer, vtkSmartPointer<vtkActor> actor)
{
	if (nullptr != actor)
	{
		actor->SetVisibility(!actor->GetVisibility());
		renderer->GetRenderWindow()->Render();
	}
}

inline void SetActorRepresentation(vtkSmartPointer<vtkRenderer> renderer, vtkSmartPointer<vtkActor> actor, Representation representation)
{
	if (nullptr != actor)
	{
		actor->GetProperty()->SetRepresentation(representation);
		renderer->GetRenderWindow()->Render();
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
		renderer->GetRenderWindow()->Render();
	}
}

class VisualDebuggingLayer
{
public:
	VisualDebuggingLayer(const string& layerName);
	~VisualDebuggingLayer();

	void Initialize(vtkSmartPointer<vtkRenderer> renderer);
	void Terminate();

	void AddPoint(const Eigen::Vector3f& p, unsigned char r, unsigned char g, unsigned char b);

	void AddLine(const Eigen::Vector3f& p0, const Eigen::Vector3f& p1, unsigned char r, unsigned char g, unsigned char b);
	
	void AddTriangle(const Eigen::Vector3f& p0, const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, unsigned char r, unsigned char g, unsigned char b);

	void AddSphere(const Eigen::Vector3f& center, const Eigen::Vector3f& scale, const Eigen::Vector3f& normal, unsigned char r, unsigned char g, unsigned char b);
	
	void AddCube(const Eigen::Vector3f& center, const Eigen::Vector3f& scale, const Eigen::Vector3f& normal, unsigned char r, unsigned char g, unsigned char b);

	void AddGlyph(const Eigen::Vector3f& center, const Eigen::Vector3f& scale, const Eigen::Vector3f& normal, unsigned char r, unsigned char g, unsigned char b);

	void AddArrow(const Eigen::Vector3f& center, const Eigen::Vector3f& normal, float scale, unsigned char r, unsigned char g, unsigned char b);

	void Update();
	void Clear();

	void ShowAll(bool show);
	void ToggleVisibilityAll();
	void SetRepresentationAll(Representation representation);
	void ToggleAllRepresentation();
	void ShowPoints(bool show);
	void TogglePoints();
	void SetRepresentationPoints(Representation representation);
	void TogglePointsRepresentation();
	void ShowLines(bool show);
	void ToggleLines();
	void SetRepresentationLines(Representation representation);
	void ToggleLinesRepresentation();
	void ShowTriangles(bool show);
	void ToggleTriangles();
	void SetRepresentationTriangles(Representation representation);
	void ToggleTrianglesRepresentation();
	void ShowSpheres(bool show);
	void ToggleSpheres();
	void SetRepresentationSpheres(Representation representation);
	void ToggleSpheresRepresentation();
	void ShowCubes(bool show);
	void ToggleCubes();
	void SetRepresentationCubes(Representation representation);
	void ToggleCubesRepresentation();
	void ShowArrows(bool show);
	void ToggleArrows();
	void SetRepresentationArrows(Representation representation);
	void ToggleArrowsRepresentation();

	float GetPointSize();
	void SetPointSize(float size);
	float GetLineWidth();
	void SetLineWidth(float width);

	inline vtkSmartPointer<vtkActor> GetPointActor() { return pointActor; }
	inline vtkSmartPointer<vtkActor> GetLineActor() { return lineActor; }
	inline vtkSmartPointer<vtkActor> GetTriangleActor() { return triangleActor; }
	inline vtkSmartPointer<vtkActor> GetSphereActor() { return sphereActor; }
	inline vtkSmartPointer<vtkActor> GetCubeActor() { return cubeActor; }
	inline vtkSmartPointer<vtkActor> GetGlyphActor() { return glyphActor; }
	inline vtkSmartPointer<vtkActor> GetArrowActor() { return arrowActor; }

private:
	string layerName = "";
	vtkSmartPointer<vtkRenderer> renderer;
	vtkSmartPointer<vtkRenderWindow> renderWindow;

	vtkSmartPointer<vtkActor> pointActor;
	vtkSmartPointer<vtkPolyDataMapper> pointPolyDataMapper;
	vtkSmartPointer<vtkPolyData> pointPolyData;

	vtkSmartPointer<vtkActor> lineActor;
	vtkSmartPointer<vtkPolyDataMapper> linePolyDataMapper;
	vtkSmartPointer<vtkPolyData> linePolyData;

	vtkSmartPointer<vtkActor> triangleActor;
	vtkSmartPointer<vtkPolyDataMapper> trianglePolyDataMapper;
	vtkSmartPointer<vtkPolyData> trianglePolyData;

	vtkSmartPointer<vtkActor> sphereActor;
	vtkSmartPointer<vtkGlyph3DMapper> spherePolyDataMapper;
	vtkSmartPointer<vtkPolyData> spherePolyData;

	vtkSmartPointer<vtkActor> cubeActor;
	vtkSmartPointer<vtkGlyph3DMapper> cubePolyDataMapper;
	vtkSmartPointer<vtkPolyData> cubePolyData;

	vtkSmartPointer<vtkActor> glyphActor;
	vtkSmartPointer<vtkGlyph3DMapper> glyphPolyDataMapper;
	vtkSmartPointer<vtkPolyData> glyphPolyData;

	vtkSmartPointer<vtkActor> arrowActor;
	vtkSmartPointer<vtkPolyDataMapper> arrowPolyDataMapper;
	vtkSmartPointer<vtkGlyph3D> arrowGlyph3D;
	vtkSmartPointer<vtkPolyData> arrowPolyData;

	void DrawPoints();
	void DrawLines();
	void DrawTriangle();
	void DrawSpheres();
	void DrawCubes();
	void DrawGlyphs();
	void DrawArrows();

	vector<std::tuple<Eigen::Vector3f, unsigned char, unsigned char, unsigned char>> pointInfosToDraw;
	vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, unsigned char, unsigned char,unsigned char>> lineInfosToDraw;
	vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f, unsigned char, unsigned char, unsigned char>> triangleInfosToDraw;
	vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f, unsigned char, unsigned char, unsigned char>> sphereInfosToDraw;
	vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f, unsigned char, unsigned char, unsigned char>> cubeInfosToDraw;
	vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f, unsigned char, unsigned char, unsigned char>> glyphInfosToDraw;
	vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, float, unsigned char, unsigned char, unsigned char>> arrowInfosToDraw;
};
