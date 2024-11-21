#include <vtkSmartPointer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkTriangle.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/VolumeToMesh.h>

int main()
{
    // Initialize OpenVDB
    openvdb::initialize();

    // Create a sphere in OpenVDB
    float radius = 50.0f;
    openvdb::Vec3f center(0.0f, 0.0f, 0.0f);
    float voxelSize = 1.0f;
    float halfWidth = 3.0f;
    openvdb::FloatGrid::Ptr sphereGrid = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(
        radius, center, voxelSize, halfWidth);

    // Convert the VDB level set into a mesh
    openvdb::tools::VolumeToMesh mesher;
    mesher(*sphereGrid);

    // Create VTK objects to store the mesh data
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> triangles = vtkSmartPointer<vtkCellArray>::New();

    // Extract points and triangles from the OpenVDB mesh
    for (size_t i = 0; i < mesher.pointListSize(); ++i) {
        const openvdb::Vec3s& p = mesher.pointList()[i];
        points->InsertNextPoint(p.x(), p.y(), p.z());
    }

    for (size_t i = 0; i < mesher.polygonPoolListSize(); ++i) {
        const openvdb::tools::PolygonPool& pool = mesher.polygonPoolList()[i];
        for (size_t j = 0; j < pool.numQuads(); ++j) {
            openvdb::Vec4I quad = pool.quad(j);
            vtkSmartPointer<vtkTriangle> triangle1 = vtkSmartPointer<vtkTriangle>::New();
            triangle1->GetPointIds()->SetId(0, quad[0]);
            triangle1->GetPointIds()->SetId(1, quad[1]);
            triangle1->GetPointIds()->SetId(2, quad[2]);
            triangles->InsertNextCell(triangle1);

            vtkSmartPointer<vtkTriangle> triangle2 = vtkSmartPointer<vtkTriangle>::New();
            triangle2->GetPointIds()->SetId(0, quad[2]);
            triangle2->GetPointIds()->SetId(1, quad[3]);
            triangle2->GetPointIds()->SetId(2, quad[0]);
            triangles->InsertNextCell(triangle2);
        }

        for (size_t j = 0; j < pool.numTriangles(); ++j) {
            openvdb::Vec3I tri = pool.triangle(j);
            vtkSmartPointer<vtkTriangle> triangle = vtkSmartPointer<vtkTriangle>::New();
            triangle->GetPointIds()->SetId(0, tri[0]);
            triangle->GetPointIds()->SetId(1, tri[1]);
            triangle->GetPointIds()->SetId(2, tri[2]);
            triangles->InsertNextCell(triangle);
        }
    }

    // Create a polydata object
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    polyData->SetPoints(points);
    polyData->SetPolys(triangles);

    // Create a mapper
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputData(polyData);

    // Create an actor
    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    // Create a renderer and render window
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);

    // Create a render window interactor
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    // Use trackball camera interactor style (for orbiting)
    vtkSmartPointer<vtkInteractorStyleTrackballCamera> style = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
    renderWindowInteractor->SetInteractorStyle(style);

    // Add the actor to the scene
    renderer->AddActor(actor);
    renderer->SetBackground(0.1, 0.2, 0.3); // Background color

    // Render and interact
    renderWindow->Render();
    renderWindowInteractor->Initialize();  // Initialize interactor
    renderWindowInteractor->Start();       // Start the interaction loop

    // Cleanup OpenVDB
    openvdb::uninitialize();

    return 0;
}
