#pragma once

#include <Debugging/VisualDebuggingLayer.h>

#include <string>
using namespace std;

class VisualDebugging {
public:
	static void Initialize(vtkSmartPointer<vtkRenderer> renderer);
	static void Terminate();

	static VisualDebuggingLayer* CreateLayer(const string& layerName);

	static void AddLine(const string& layerName, const Eigen::Vector3f& p0,
		const Eigen::Vector3f& p1, unsigned char r, unsigned char g,
		unsigned char b);

	static void AddTriangle(const string& layerName, const Eigen::Vector3f& p0,
		const Eigen::Vector3f& p1, const Eigen::Vector3f& p2,
		unsigned char r, unsigned char g, unsigned char b);

	static void AddSphere(const string& layerName, const Eigen::Vector3f& center, const Eigen::Vector3f& scale, const Eigen::Vector3f& normal, unsigned char r, unsigned char g, unsigned char b);

	static void AddCube(const string& layerName, const Eigen::Vector3f& center, const Eigen::Vector3f& scale, const Eigen::Vector3f& normal, unsigned char r, unsigned char g, unsigned char b);

	static void AddGlyph(const string& layerName, const Eigen::Vector3f& center, const Eigen::Vector3f& scale, const Eigen::Vector3f& normal, unsigned char r, unsigned char g, unsigned char b);

	static void AddArrow(const string& layerName, const Eigen::Vector3f& center, const Eigen::Vector3f& normal, float scale, unsigned char r, unsigned char g, unsigned char b);

	static void Update();
	static void ClearAll();
	static void Clear(const string& layerName);

	static void ToggleVisibilityAll();
	static void ToggleVisibility(const string& layerName);
	static void ToggleVisibilityByIndex(int index);
	static void SetRepresentationAll(Representation representation);
	static void SetRepresentation(const string& layerName, Representation representation);
	static void SetRepresentationByIndex(int index, Representation representation);
	static void ToggleRepresentationAll();
	static void ToggleRepresentation(const string& layerName);
	static void ToggleRepresentationByIndex(int index);

	static float GetPointSize(const string& layerName);
	static void SetPointSize(const string& layerName, float size);
	static float GetLineWidth(const string& layerName);
	static void SetLineWidth(const string& layerName, float width);

	static vtkSmartPointer<vtkActor> GetPointActor(const string& layerName);
	static vtkSmartPointer<vtkActor> GetLineActor(const string& layerName);
	static vtkSmartPointer<vtkActor> GetTriangleActor(const string& layerName);
	static vtkSmartPointer<vtkActor> GetSphereActor(const string& layerName);
	static vtkSmartPointer<vtkActor> GetCubeActor(const string& layerName);
	static vtkSmartPointer<vtkActor> GetGlyphActor(const string& layerName);
	static vtkSmartPointer<vtkActor> GetArrowActor(const string& layerName);

	static int GetNumberOfLayers();

private:
	VisualDebugging();
	~VisualDebugging();

	static bool s_needToRender;
	static VisualDebugging* s_instance;

	static map<string, int> s_layerNameIndexMapping;
	static vector<VisualDebuggingLayer*> s_layers;

	static vtkSmartPointer<vtkRenderer> s_renderer;
	static vtkSmartPointer<vtkRenderWindow> s_renderWindow;

	static VisualDebuggingLayer* GetLayer(const string& layerName);
};
