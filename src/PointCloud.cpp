#include <openvdb/openvdb.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <vtkSmartPointer.h>
#include <vtkPLYReader.h>
#include <vtkPLYWriter.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkProperty.h>
#include <vtkTriangle.h>
#include <vtkVertexGlyphFilter.h>

#include <iostream>
#include <vector>
#include <string>
#include <sstream>

vtkSmartPointer<vtkPolyData> readPLY(const std::string& filePath) {
    vtkSmartPointer<vtkPLYReader> reader = vtkSmartPointer<vtkPLYReader>::New();
    reader->SetFileName(filePath.c_str());
    reader->Update();
    return reader->GetOutput();
}

void writePLY(vtkSmartPointer<vtkPolyData> data, const std::string& filePath) {
    vtkSmartPointer<vtkPLYWriter> writer = vtkSmartPointer<vtkPLYWriter>::New();
    writer->SetFileName(filePath.c_str());
    writer->SetInputData(data);
    writer->Update();
}

std::vector<openvdb::Vec3s> polyDataToVector(vtkSmartPointer<vtkPolyData> polyData) {
    std::vector<openvdb::Vec3s> points;
    vtkPoints* vtkPoints = polyData->GetPoints();
    points.reserve(vtkPoints->GetNumberOfPoints());

    for (vtkIdType i = 0; i < vtkPoints->GetNumberOfPoints(); ++i) {
        double point[3];
        vtkPoints->GetPoint(i, point);
        points.emplace_back(point[0], point[1], point[2]);
    }

    return points;
}

openvdb::FloatGrid::Ptr createOpenVDBVolume(const std::vector<openvdb::Vec3s>& points, float voxelSize = 0.1f) {
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(0.0f);
    grid->setTransform(openvdb::math::Transform::createLinearTransform(voxelSize));

    openvdb::FloatGrid::Accessor accessor = grid->getAccessor();

    for (const auto& point : points) {
        openvdb::Coord xyz = grid->transform().worldToIndexNodeCentered(point);
        accessor.setValue(xyz, 1.0f);
    }

    return grid;
}

vtkSmartPointer<vtkPolyData> openVDBToMesh(openvdb::FloatGrid::Ptr grid) {
    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> triangles;
    std::vector<openvdb::Vec4I> quads;

    // Convert VDB grid to mesh
    double isovalue = 0.5;
    double adaptivity = 0.0;
    openvdb::tools::volumeToMesh(*grid, points, triangles, quads, isovalue, adaptivity);

    // Create VTK points
    vtkSmartPointer<vtkPoints> vtkPts = vtkSmartPointer<vtkPoints>::New();
    for (const auto& point : points) {
        vtkPts->InsertNextPoint(point.x(), point.y(), point.z());
    }

    // Create VTK cells
    vtkSmartPointer<vtkCellArray> vtkPolys = vtkSmartPointer<vtkCellArray>::New();
    for (const auto& triangle : triangles) {
        vtkPolys->InsertNextCell(3);
        vtkPolys->InsertCellPoint(triangle[0]);
        vtkPolys->InsertCellPoint(triangle[1]);
        vtkPolys->InsertCellPoint(triangle[2]);
    }
    for (const auto& quad : quads) {
        vtkPolys->InsertNextCell(4);
        vtkPolys->InsertCellPoint(quad[0]);
        vtkPolys->InsertCellPoint(quad[1]);
        vtkPolys->InsertCellPoint(quad[2]);
        vtkPolys->InsertCellPoint(quad[3]);
    }

    // Create VTK polydata
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    polyData->SetPoints(vtkPts);
    polyData->SetPolys(vtkPolys);

    return polyData;
}

#define SUB(a, b) ((a) - (b))
#define SQUARE(a) ((a) * (a))
#define DISTANCE(p0, p1) sqrtf(SQUARE(SUB(p1[0], p0[0])) + SQUARE(SUB(p1[1], p0[1])) + SQUARE(SUB(p1[2], p0[2])))

int main() {
    openvdb::initialize();

    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);
    vtkSmartPointer<vtkRenderWindowInteractor> interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    vtkNew<vtkInteractorStyleTrackballCamera> trackballStyle;
    interactor->SetInteractorStyle(trackballStyle);
    interactor->SetRenderWindow(renderWindow);

    // 400 x 480
    int patch_width = 400;
    int patch_height = 480;
    int half_patch_width = patch_width / 2;
    int half_patch_height = patch_height / 2;
    float xUnit = 0.1f;
    float yUnit = 0.1f;

    vtkNew<vtkPoints> planePoints;
    for (int y = -half_patch_height; y <= half_patch_height; y++)
    {
        for (int x = -half_patch_width; x <= half_patch_width; x++)
        {
            planePoints->InsertNextPoint((float)x * xUnit, (float)y * yUnit, FLT_MAX);
        }
    }

    vtkNew<vtkPolyData> polyData;
    polyData->SetPoints(planePoints);

    for (size_t i = 1; i < 2; i++)
    {
        std::stringstream ss;
        ss << "C:\\Resources\\Debug\\HD\\patch_" << i << ".ply";
        vtkSmartPointer<vtkPolyData> plyPolyData = readPLY(ss.str());

        auto points = plyPolyData->GetPoints();
        for (size_t i = 0; i < plyPolyData->GetNumberOfPoints(); i++)
        {
            auto point = points->GetPoint(i);
            float x = point[0];
            float y = point[1];
            float z = point[2];

            //printf("%f, %f, %f\n", x, y, z);

            auto xi = (int)roundf(x / xUnit) + half_patch_width;
            auto yi = (int)roundf(y / yUnit) + half_patch_height;

            auto p = planePoints->GetPoint(yi * (patch_width + 1) + xi);
            p[2] = z;
            //printf("z : %f\n", z);
            planePoints->SetPoint(yi * (patch_width + 1) + xi, p);
        }

        {
            vtkSmartPointer<vtkVertexGlyphFilter> vertexFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
            vertexFilter->SetInputData(plyPolyData);
            vertexFilter->Update();

            vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
            mapper->SetInputConnection(vertexFilter->GetOutputPort());

            vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
            actor->SetMapper(mapper);
            actor->GetProperty()->SetColor(1.0, 1.0, 1.0);

            actor->GetProperty()->SetPointSize(5);
            actor->GetProperty()->SetColor(1.0, 0.0, 0.0);

            renderer->AddActor(actor);
            renderer->SetBackground(0.3, 0.5, 0.7);
        }
    }

    vtkNew<vtkCellArray> triangles;
    for (int y = 0; y < patch_height; y++)
    {
        for (int x = 0; x < patch_width; x++)
        {
            vtkIdType pi00 = y * (patch_width + 1) + x;
            vtkIdType pi01 = y * (patch_width + 1) + (x + 1);
            vtkIdType pi10 = (y + 1) * (patch_width + 1) + x;
            vtkIdType pi11 = (y + 1) * (patch_width + 1) + (x + 1);

            auto p00 = planePoints->GetPoint(pi00);
            auto p01 = planePoints->GetPoint(pi01);
            auto p10 = planePoints->GetPoint(pi10);
            auto p11 = planePoints->GetPoint(pi11);

            if (0.2f < DISTANCE(p00, p01))
            {

            }

            // First triangle (lower-left)
            vtkIdType triangle0[3] = { pi00, pi01, pi11 };
            triangles->InsertNextCell(3, triangle0);

            // Second triangle (upper-right)
            vtkIdType triangle1[3] = { pi00, pi11, pi10 };
            triangles->InsertNextCell(3, triangle1);
        }
    }
    polyData->SetPolys(triangles);

    writePLY(polyData, "C:\\Resources\\Debug\\HD\\patch_mesh_0.ply");

    {
        vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInputData(polyData);

        vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
        actor->SetMapper(mapper);
        actor->GetProperty()->SetColor(1.0, 1.0, 1.0);

        renderer->AddActor(actor);
        renderer->SetBackground(0.3, 0.5, 0.7);
    }

    renderWindow->Render();
    interactor->Start();

    return 0;
}
