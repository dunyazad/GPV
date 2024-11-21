//#include <Common.h>
//
//#include <openvdb/openvdb.h>
//
//#include <Eigen/Dense>
//#include <Eigen/Sparse>
//#include <Eigen/IterativeLinearSolvers>
//
//#include <App/CustomTrackballStyle.h>
//#include <App/Utility.h>
//
//#include <Algorithm/vtkMedianFilter.h>
//#include <Algorithm/vtkQuantizingFilter.h>
//
//#include <Debugging/VisualDebugging.h>
//
//#include <CUDA/KDTree.cuh>
//#include <CUDA/SVO.cuh>
//
//#include <pcl/point_types.h>
//#include <pcl/io/pcd_io.h>
//#include <pcl/point_cloud.h>
//#include <pcl/features/normal_3d.h>
//#include <pcl/kdtree/kdtree_flann.h>
//#include <pcl/surface/marching_cubes.h>
//#include <pcl/surface/marching_cubes_hoppe.h>
//#include <pcl/octree/octree.h>
//#include <pcl/io/ply_io.h> // For saving PLY files
//#include <pcl/io/pcd_io.h> // For saving and loading PCD files
////#include <pcl/surface/greedy_projection.h>
//
//int pid = 0;
//size_t size_0 = 0;
//size_t size_45 = 0;
//float transform_0[16];
//float transform_45[16];
//unsigned char image_0[400 * 480];
//unsigned char image_45[400 * 480];
//Eigen::Vector3f points_0[400 * 480];
//Eigen::Vector3f points_45[400 * 480];
//
//int index = 0;
//
//
//
//
//
//
//// Definition of the Octree Node
//struct OctreeNode {
//	std::vector<Eigen::Vector3f> voxels;  // Stores points (voxels) within this node
//	std::unique_ptr<OctreeNode> children[8];  // Each node can have up to 8 children
//	bool isLeaf;
//
//	OctreeNode() : isLeaf(true) {}
//};
//
//class SparseVoxelOctree {
//public:
//	SparseVoxelOctree(Eigen::Vector3f origin, float halfSize)
//		: root(std::make_unique<OctreeNode>()), origin(origin), halfSize(halfSize) {}
//
//	void insert(const Eigen::Vector3f& point) {
//		insert(root.get(), point, origin, halfSize);
//	}
//
//	void print() const {
//		print(root.get(), origin, halfSize);
//	}
//
//	//vtkSmartPointer<vtkActor> visualize() const {
//	//	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
//	//	collectPoints(root.get(), origin, halfSize, points);
//
//	//	vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
//	//	polyData->SetPoints(points);
//
//	//	vtkSmartPointer<vtkVertexGlyphFilter> vertexFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
//	//	vertexFilter->SetInputData(polyData);
//	//	vertexFilter->Update();
//
//	//	vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
//	//	mapper->SetInputConnection(vertexFilter->GetOutputPort());
//
//	//	vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
//	//	actor->SetMapper(mapper);
//
//	//	return actor;
//	//}
//
//	void visualize(vtkSmartPointer<vtkRenderer> renderer) const {
//		vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
//		collectPoints(root.get(), origin, halfSize, points);
//
//		vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
//		polyData->SetPoints(points);
//
//		vtkSmartPointer<vtkVertexGlyphFilter> vertexFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
//		vertexFilter->SetInputData(polyData);
//		vertexFilter->Update();
//
//		vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
//		mapper->SetInputConnection(vertexFilter->GetOutputPort());
//
//		vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
//		actor->SetMapper(mapper);
//
//		//renderer->AddActor(actor);
//
//		//visualizeNodeWireframes(root.get(), origin, halfSize, renderer);
//
//		visualizeMesh(points, renderer);
//	}
//
//	void visualizeNodeWireframes(const OctreeNode* node, Eigen::Vector3f nodeOrigin, float nodeHalfSize, vtkSmartPointer<vtkRenderer> renderer) const {
//		if (!node) return;
//
//		if (node->isLeaf && node->voxels.size() != 0)
//		{
//			Eigen::Vector3f minCorner = nodeOrigin - Eigen::Vector3f(nodeHalfSize, nodeHalfSize, nodeHalfSize);
//			Eigen::Vector3f maxCorner = nodeOrigin + Eigen::Vector3f(nodeHalfSize, nodeHalfSize, nodeHalfSize);
//
//			vtkSmartPointer<vtkPoints> cubePoints = vtkSmartPointer<vtkPoints>::New();
//			cubePoints->InsertNextPoint(minCorner.x(), minCorner.y(), minCorner.z());
//			cubePoints->InsertNextPoint(maxCorner.x(), minCorner.y(), minCorner.z());
//			cubePoints->InsertNextPoint(maxCorner.x(), maxCorner.y(), minCorner.z());
//			cubePoints->InsertNextPoint(minCorner.x(), maxCorner.y(), minCorner.z());
//			cubePoints->InsertNextPoint(minCorner.x(), minCorner.y(), maxCorner.z());
//			cubePoints->InsertNextPoint(maxCorner.x(), minCorner.y(), maxCorner.z());
//			cubePoints->InsertNextPoint(maxCorner.x(), maxCorner.y(), maxCorner.z());
//			cubePoints->InsertNextPoint(minCorner.x(), maxCorner.y(), maxCorner.z());
//
//			vtkSmartPointer<vtkCellArray> edges = vtkSmartPointer<vtkCellArray>::New();
//			int edgeIndices[12][2] = {
//				{0, 1}, {1, 2}, {2, 3}, {3, 0},
//				{4, 5}, {5, 6}, {6, 7}, {7, 4},
//				{0, 4}, {1, 5}, {2, 6}, {3, 7}
//			};
//
//			for (auto& edge : edgeIndices) {
//				vtkSmartPointer<vtkLine> line = vtkSmartPointer<vtkLine>::New();
//				line->GetPointIds()->SetId(0, edge[0]);
//				line->GetPointIds()->SetId(1, edge[1]);
//				edges->InsertNextCell(line);
//			}
//
//			vtkSmartPointer<vtkUnstructuredGrid> grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
//			grid->SetPoints(cubePoints);
//			grid->SetCells(VTK_LINE, edges);
//
//			vtkSmartPointer<vtkDataSetMapper> wireframeMapper = vtkSmartPointer<vtkDataSetMapper>::New();
//			wireframeMapper->SetInputData(grid);
//
//			vtkSmartPointer<vtkActor> wireframeActor = vtkSmartPointer<vtkActor>::New();
//			wireframeActor->SetMapper(wireframeMapper);
//			wireframeActor->GetProperty()->SetColor(1.0, 1.0, 1.0);
//			wireframeActor->GetProperty()->SetLineWidth(1.5);
//
//			renderer->AddActor(wireframeActor);
//		}
//
//		if (!node->isLeaf) {
//			for (int i = 0; i < 8; ++i) {
//				if (node->children[i]) {
//					visualizeNodeWireframes(node->children[i].get(), getChildOrigin(nodeOrigin, nodeHalfSize, i), nodeHalfSize / 2, renderer);
//				}
//			}
//		}
//	}
//
//	/*void visualizeMesh(vtkSmartPointer<vtkPoints> points, vtkSmartPointer<vtkRenderer> renderer) const {
//		vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
//		polyData->SetPoints(points);
//
//		vtkSmartPointer<vtkMarchingCubes> marchingCubes = vtkSmartPointer<vtkMarchingCubes>::New();
//		marchingCubes->SetInputData(polyData);
//		marchingCubes->ComputeNormalsOn();
//		marchingCubes->SetValue(0, 0.0);
//		marchingCubes->Update();
//
//		vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
//		mapper->SetInputConnection(marchingCubes->GetOutputPort());
//
//		vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
//		actor->SetMapper(mapper);
//		actor->GetProperty()->SetColor(0.8, 0.8, 0.8);
//		actor->GetProperty()->SetOpacity(0.5);
//
//		renderer->AddActor(actor);
//	}*/
//	/*void visualizeMesh(vtkSmartPointer<vtkPoints> points, vtkSmartPointer<vtkRenderer> renderer) const {
//		vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
//		imageData->SetDimensions(50, 50, 50);
//		imageData->SetOrigin(-25, -25, -25);
//		imageData->SetSpacing(0.1, 0.1, 0.1);
//		imageData->AllocateScalars(VTK_UNSIGNED_CHAR, 1);
//
//		unsigned char* imageDataPtr = static_cast<unsigned char*>(imageData->GetScalarPointer());
//		std::fill(imageDataPtr, imageDataPtr + imageData->GetDimensions()[0] * imageData->GetDimensions()[1] * imageData->GetDimensions()[2], 0);
//
//		for (vtkIdType i = 0; i < points->GetNumberOfPoints(); ++i) {
//			double point[3];
//			points->GetPoint(i, point);
//			int x = static_cast<int>(point[0] + 25);
//			int y = static_cast<int>(point[1] + 25);
//			int z = static_cast<int>(point[2] + 25);
//			if (x >= 0 && x < 50 && y >= 0 && y < 50 && z >= 0 && z < 50) {
//				unsigned char* voxel = static_cast<unsigned char*>(imageData->GetScalarPointer(x, y, z));
//				*voxel = 255;
//			}
//		}
//
//		vtkSmartPointer<vtkMarchingCubes> marchingCubes = vtkSmartPointer<vtkMarchingCubes>::New();
//		marchingCubes->SetInputData(imageData);
//		marchingCubes->ComputeNormalsOn();
//		marchingCubes->SetValue(0, 127.5);
//		marchingCubes->Update();
//
//		vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
//		mapper->SetInputConnection(marchingCubes->GetOutputPort());
//
//		vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
//		actor->SetMapper(mapper);
//		actor->GetProperty()->SetColor(0.8, 0.8, 0.8);
//		actor->GetProperty()->SetOpacity(0.5);
//
//		renderer->AddActor(actor);
//	}*/
//	//void visualizeMesh(vtkSmartPointer<vtkPoints> points, vtkSmartPointer<vtkRenderer> renderer) const {
//	//	vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
//	//	imageData->SetDimensions(500, 500, 500);
//	//	imageData->SetOrigin(-25, -25, -25);
//	//	imageData->SetSpacing(1.0, 1.0, 1.0);
//	//	imageData->AllocateScalars(VTK_UNSIGNED_CHAR, 1);
//
//	//	unsigned char* imageDataPtr = static_cast<unsigned char*>(imageData->GetScalarPointer());
//	//	std::fill(imageDataPtr, imageDataPtr + imageData->GetDimensions()[0] * imageData->GetDimensions()[1] * imageData->GetDimensions()[2], 0);
//
//	//	for (vtkIdType i = 0; i < points->GetNumberOfPoints(); ++i) {
//	//		double point[3];
//	//		points->GetPoint(i, point);
//	//		int x = static_cast<int>(point[0] + 25);
//	//		int y = static_cast<int>(point[1] + 25);
//	//		int z = static_cast<int>(point[2] + 25);
//	//		if (x >= 0 && x < 50 && y >= 0 && y < 50 && z >= 0 && z < 50) {
//	//			unsigned char* voxel = static_cast<unsigned char*>(imageData->GetScalarPointer(x, y, z));
//	//			*voxel = 255;
//	//		}
//	//	}
//
//	//	vtkSmartPointer<vtkMarchingCubes> marchingCubes = vtkSmartPointer<vtkMarchingCubes>::New();
//	//	marchingCubes->SetInputData(imageData);
//	//	marchingCubes->ComputeNormalsOn();
//	//	marchingCubes->SetValue(0, 127.5);
//	//	marchingCubes->Update();
//
//	//	// Apply decimation to reduce the number of triangles and potentially remove inside faces
//	//	vtkSmartPointer<vtkDecimatePro> decimate = vtkSmartPointer<vtkDecimatePro>::New();
//	//	decimate->SetInputConnection(marchingCubes->GetOutputPort());
//	//	decimate->SetTargetReduction(0.5);  // Reduce by 50%
//	//	decimate->PreserveTopologyOn();
//	//	decimate->Update();
//
//	//	// Apply normals to ensure the normals are consistent and improve surface quality
//	//	vtkSmartPointer<vtkPolyDataNormals> normals = vtkSmartPointer<vtkPolyDataNormals>::New();
//	//	normals->SetInputConnection(decimate->GetOutputPort());
//	//	normals->ConsistencyOn();
//	//	normals->SplittingOff();
//	//	normals->Update();
//
//	//	vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
//	//	mapper->SetInputConnection(normals->GetOutputPort());
//
//	//	vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
//	//	actor->SetMapper(mapper);
//	//	actor->GetProperty()->SetColor(0.8, 0.8, 0.8);
//	//	actor->GetProperty()->SetOpacity(0.5);
//
//	//	renderer->AddActor(actor);
//	//}
//
//void visualizeMesh(vtkSmartPointer<vtkPoints> points, vtkSmartPointer<vtkRenderer> renderer) const {
//	vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
//	imageData->SetDimensions(50, 50, 50);
//	imageData->SetOrigin(-25, -25, -25);
//	imageData->SetSpacing(1.0, 1.0, 1.0);
//	imageData->AllocateScalars(VTK_UNSIGNED_CHAR, 1);
//
//	unsigned char* imageDataPtr = static_cast<unsigned char*>(imageData->GetScalarPointer());
//	std::fill(imageDataPtr, imageDataPtr + imageData->GetDimensions()[0] * imageData->GetDimensions()[1] * imageData->GetDimensions()[2], 0);
//
//	for (vtkIdType i = 0; i < points->GetNumberOfPoints(); ++i) {
//		double point[3];
//		points->GetPoint(i, point);
//		int x = static_cast<int>(point[0] + 25);
//		int y = static_cast<int>(point[1] + 25);
//		int z = static_cast<int>(point[2] + 25);
//		if (x >= 0 && x < 50 && y >= 0 && y < 50 && z >= 0 && z < 50) {
//			unsigned char* voxel = static_cast<unsigned char*>(imageData->GetScalarPointer(x, y, z));
//			*voxel = 255;
//		}
//	}
//
//	vtkSmartPointer<vtkMarchingCubes> marchingCubes = vtkSmartPointer<vtkMarchingCubes>::New();
//	marchingCubes->SetInputData(imageData);
//	marchingCubes->ComputeNormalsOn();
//	marchingCubes->SetValue(0, 127.5);
//	marchingCubes->Update();
//
//	vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
//	mapper->SetInputConnection(marchingCubes->GetOutputPort());
//
//	vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
//	actor->SetMapper(mapper);
//	actor->GetProperty()->SetColor(0.8, 0.8, 0.8);
//	actor->GetProperty()->SetOpacity(0.5);
//	actor->GetProperty()->BackfaceCullingOn(); // Enable backface culling to remove backside
//
//	renderer->AddActor(actor);
//}
//
//
//
//
//private:
//	std::unique_ptr<OctreeNode> root;
//	Eigen::Vector3f origin;
//	float halfSize;
//
//	void insert(OctreeNode* node, const Eigen::Vector3f& point, Eigen::Vector3f nodeOrigin, float nodeHalfSize) {
//		if (node->isLeaf) {
//			if (node->voxels.size() < MAX_VOXELS_PER_NODE) {
//				node->voxels.push_back(point);
//			}
//			else {
//				subdivide(node, nodeOrigin, nodeHalfSize);
//				// Reinsert existing voxels into children
//				for (const auto& voxel : node->voxels) {
//					int childIndex = getChildIndex(nodeOrigin, nodeHalfSize, voxel);
//					insert(node->children[childIndex].get(), voxel, getChildOrigin(nodeOrigin, nodeHalfSize, childIndex), nodeHalfSize / 2);
//				}
//				node->voxels.clear();
//				node->isLeaf = false;
//				// Insert the new point
//				int childIndex = getChildIndex(nodeOrigin, nodeHalfSize, point);
//				insert(node->children[childIndex].get(), point, getChildOrigin(nodeOrigin, nodeHalfSize, childIndex), nodeHalfSize / 2);
//			}
//		}
//		else {
//			int childIndex = getChildIndex(nodeOrigin, nodeHalfSize, point);
//			insert(node->children[childIndex].get(), point, getChildOrigin(nodeOrigin, nodeHalfSize, childIndex), nodeHalfSize / 2);
//		}
//	}
//
//	void subdivide(OctreeNode* node, Eigen::Vector3f nodeOrigin, float nodeHalfSize) {
//		for (int i = 0; i < 8; ++i) {
//			node->children[i] = std::make_unique<OctreeNode>();
//		}
//	}
//
//	int getChildIndex(const Eigen::Vector3f& nodeOrigin, float nodeHalfSize, const Eigen::Vector3f& point) const {
//		int index = 0;
//		if (point.x() >= nodeOrigin.x()) index |= 1;
//		if (point.y() >= nodeOrigin.y()) index |= 2;
//		if (point.z() >= nodeOrigin.z()) index |= 4;
//		return index;
//	}
//
//	Eigen::Vector3f getChildOrigin(const Eigen::Vector3f& nodeOrigin, float nodeHalfSize, int childIndex) const {
//		float offset = nodeHalfSize / 2;
//		return Eigen::Vector3f(
//			nodeOrigin.x() + ((childIndex & 1) ? offset : -offset),
//			nodeOrigin.y() + ((childIndex & 2) ? offset : -offset),
//			nodeOrigin.z() + ((childIndex & 4) ? offset : -offset)
//		);
//	}
//
//	void print(const OctreeNode* node, Eigen::Vector3f nodeOrigin, float nodeHalfSize) const {
//		if (node->isLeaf) {
//			for (const auto& voxel : node->voxels) {
//				std::cout << "Voxel at: (" << voxel.x() << ", " << voxel.y() << ", " << voxel.z() << ")\n";
//			}
//		}
//		else {
//			for (int i = 0; i < 8; ++i) {
//				if (node->children[i]) {
//					print(node->children[i].get(), getChildOrigin(nodeOrigin, nodeHalfSize, i), nodeHalfSize / 2);
//				}
//			}
//		}
//	}
//
//	void collectPoints(const OctreeNode* node, Eigen::Vector3f nodeOrigin, float nodeHalfSize, vtkSmartPointer<vtkPoints> points) const {
//		if (node->isLeaf) {
//			for (const auto& voxel : node->voxels) {
//				points->InsertNextPoint(voxel.x(), voxel.y(), voxel.z());
//			}
//		}
//		else {
//			for (int i = 0; i < 8; ++i) {
//				if (node->children[i]) {
//					collectPoints(node->children[i].get(), getChildOrigin(nodeOrigin, nodeHalfSize, i), nodeHalfSize / 2, points);
//				}
//			}
//		}
//	}
//
//	static constexpr int MAX_VOXELS_PER_NODE = 1;  // Threshold for subdivision
//};
//
//
//
//
//
//
//
//
//
//void OnKeyPress(vtkObject* caller, long unsigned int eventId, void* clientData, void* callData) {
//    vtkRenderWindowInteractor* interactor = static_cast<vtkRenderWindowInteractor*>(caller);
//    std::string key = interactor->GetKeySym();
//
//    printf("%s\n", key.c_str());
//
//    if (key == "r")
//	{
//        std::cout << "Key 'r' was pressed. Resetting camera." << std::endl;
//        vtkRenderer* renderer = static_cast<vtkRenderer*>(clientData);
//
//        vtkCamera* camera = renderer->GetActiveCamera();
//        renderer->ResetCamera();
//
//        // Reset the camera position, focal point, and view up vector to their defaults
//        camera->SetPosition(0, 0, 1);          // Reset position to default
//        camera->SetFocalPoint(0, 0, 0);        // Reset focal point to origin
//        camera->SetViewUp(0, 1, 0);            // Reset the up vector to default (Y-axis up)
//        
//		interactor->Render();
//    }
//	else if (key == "c")
//	{
//		vtkRenderer* renderer = static_cast<vtkRenderer*>(clientData);
//
//		vtkCamera* camera = renderer->GetActiveCamera();
//		camera->SetParallelProjection(!camera->GetParallelProjection());
//
//		interactor->Render();
//	}
//    else if (key == "Escape")
//	{
//        std::cout << "Key 'Escape' was pressed. Exiting." << std::endl;
//        interactor->TerminateApp();
//    }
//	else if (key == "space")
//	{
//		VisualDebugging::SetLineWidth("Spheres", 1);
//		vtkSmartPointer<vtkActor> actor = VisualDebugging::GetSphereActor("Spheres");
//		vtkSmartPointer<vtkMapper> mapper = actor->GetMapper();
//		vtkSmartPointer<vtkPolyDataMapper> polyDataMapper =
//			vtkPolyDataMapper::SafeDownCast(mapper);
//		vtkSmartPointer<vtkGlyph3DMapper> glyph3DMapper = vtkGlyph3DMapper::SafeDownCast(mapper);
//		vtkSmartPointer<vtkPolyData> polyData = vtkPolyData::SafeDownCast(glyph3DMapper->GetInputDataObject(0, 0));
//		vtkSmartPointer<vtkPointData> pointData = polyData->GetPointData();
//		vtkSmartPointer<vtkDoubleArray> scaleArray =
//			vtkDoubleArray::SafeDownCast(pointData->GetArray("Scales"));
//		for (vtkIdType i = 0; i < scaleArray->GetNumberOfTuples(); ++i)
//		{
//			double scale[3]; // Assuming 3-component scale array (X, Y, Z)
//			scaleArray->GetTuple(i, scale);
//			//std::cout << "Scale for point " << i << ": "
//			//	<< scale[0 ] << ", " << scale[1] << ", " << scale[2] << std::endl;
//			scale[0] *= 0.9;
//			scale[1] *= 0.9;
//			scale[2] *= 0.9;
//			scaleArray->SetTuple(i, scale);
//		}
//		polyData->Modified();
//		glyph3DMapper->SetScaleArray("Scales");
//		glyph3DMapper->Update();
//	}
//	else if (key == "Return")
//	{
//		vtkRenderer* renderer = static_cast<vtkRenderer*>(clientData);
//		vtkRenderWindow* renderWindow = renderer->GetRenderWindow();
//
//		// Grab the depth buffer (z-values)
//		int width = renderWindow->GetSize()[0];
//		int height = renderWindow->GetSize()[1];
//		vtkSmartPointer<vtkFloatArray> depthBuffer = vtkSmartPointer<vtkFloatArray>::New();
//		depthBuffer->SetNumberOfComponents(1);
//		depthBuffer->SetNumberOfTuples(width * height);
//
//		renderWindow->GetZbufferData(0, 0, width - 1, height - 1, depthBuffer);
//
//		// Save depth map to an image
//		float minDepth = std::numeric_limits<float>::max();
//		float maxDepth = std::numeric_limits<float>::lowest();
//
//		for (vtkIdType i = 0; i < depthBuffer->GetNumberOfTuples(); i++) {
//			float depthValue = depthBuffer->GetValue(i);
//			minDepth = std::min(minDepth, depthValue);
//			maxDepth = std::max(maxDepth, depthValue);
//		}
//		
//		vtkSmartPointer<vtkImageData> depthImage = vtkSmartPointer<vtkImageData>::New();
//		depthImage->SetDimensions(width, height, 1);
//		depthImage->AllocateScalars(VTK_FLOAT, 1);
//
//		for (int y = 0; y < height; ++y)
//		{
//			for (int x = 0; x < width; ++x)
//			{
//				float depthValue = depthBuffer->GetValue((height - y - 1) * width + x);
//				float normalizedDepth = (depthValue - minDepth) / (maxDepth - minDepth);
//
//				depthImage->SetScalarComponentFromFloat(x, y, 0, 0, normalizedDepth * 255);
//
//				if (0.0f != normalizedDepth)
//				{
//					VisualDebugging::AddSphere("Depth", { (float)x / (float)width * 100.0f, (float)y / (float)height * 100.0f, normalizedDepth * 100.0f }, { 0.1f, 0.1f, 0.1f }, { 0.0f, 0.0f, 0.0f }, 255, 255, 255);
//				}
//			}
//		}
//
//		// Optional: Cast the float image to unsigned char to save as PNG
//		vtkSmartPointer<vtkImageCast> castFilter = vtkSmartPointer<vtkImageCast>::New();
//		castFilter->SetInputData(depthImage);
//		castFilter->SetOutputScalarTypeToUnsignedChar();
//		castFilter->Update();
//
//		vtkSmartPointer<vtkPNGWriter> writer = vtkSmartPointer<vtkPNGWriter>::New();
//		writer->SetFileName("c:\\Debug\\2D\\depthmap.png");
//		writer->SetInputData(castFilter->GetOutput());
//		writer->Write();
//	}
//	else if (key == "1")
//	{
//		VisualDebugging::ToggleVisibility("Spheres");
//	}
//	else if (key == "Right")
//	{
//	/*	for (size_t i = 0; i < 10; i++)
//		{
//			int h = index / 256;
//			int w = index % 256;
//
//			float x = (float)(w - 128) * 0.01f;
//			float y = (float)(h - 80) * 0.01f;
//
//			auto& p = points_0[(h * 3) * 256 + 255 - w] * 0.1f;
//
//			VisualDebugging::AddSphere("patch", { x, y, p.z() }, { 0.01f, 0.01f, 0.01f }, { 0.0f, 0.0f, 0.0f }, 0, 0, 255);
//
//			index++;
//		}*/
//
//		for (size_t i = 0; i < 100; i++)
//		{
//			Eigen::Vector3f p = points_0[index];
//			while (p.x() == FLT_MAX || p.y() == FLT_MAX || p.z() == FLT_MAX)
//			{
//				index++;
//				p = points_0[index];
//			}
//			VisualDebugging::AddSphere("patch", p, { 0.1f, 0.1f, 0.1f }, { 0.0f, 0.0f, 0.0f }, 0, 0, 255);
//
//			printf("%f, %f, %f\n", p.x(), p.y(), p.z());
//
//			index++;
//		}
//	}
//}
//
//class TimerCallback : public vtkCommand
//{
//public:
//    static TimerCallback* New() { return new TimerCallback; }
//
//    TimerCallback() = default;
//
//    virtual void Execute(vtkObject* caller, unsigned long eventId, void* vtkNotUsed(callData)) override
//    {
//        if (eventId == vtkCommand::TimerEvent) {
//            animate();
//        }
//        else {
//            std::cerr << "Unexpected event ID: " << eventId << std::endl;
//        }
//    }
//
//private:
//    void animate() { VisualDebugging::Update(); }
//};
//
//void LoadPatch(int patchID, vtkRenderer* renderer)
//{
//	stringstream ss;
//	ss << "C:\\Debug\\Patches\\patch_" << patchID << ".pat";
//
//	ifstream ifs;
//	ifs.open(ss.str(), ios::in | ios::binary);
//
//	ifs.read((char*)&pid, sizeof(int));
//	ifs.read((char*)&size_0, sizeof(size_t));
//	ifs.read((char*)&size_45, sizeof(size_t));
//	ifs.read((char*)&transform_0, sizeof(float) * 16);
//	ifs.read((char*)&transform_45, sizeof(float) * 16);
//	ifs.read((char*)&image_0, sizeof(unsigned char) * 400 * 480);
//	ifs.read((char*)&image_45, sizeof(unsigned char) * 400 * 480);
//	ifs.read((char*)&points_0, sizeof(Eigen::Vector3f) * size_0);
//	ifs.read((char*)&points_45, sizeof(Eigen::Vector3f) * size_45);
//
//	ifs.close();
//
//	vtkNew<vtkPoints> points;
//	//points->SetNumberOfPoints(size_0);
//
//	SparseVoxelOctree svo(Eigen::Vector3f(0, 0, 0), 50.0f);
//
//	int count = 0;
//	for (size_t i = 0; i < size_0; i++)
//	{
//		auto& p = points_0[i];
//
//		if ((p.x() > -1000 && p.y() > -1000 && p.z() > -1000) &&
//			(p.x() < 1000 && p.y() < 1000 && p.z() < 1000))
//		{
//			points->InsertNextPoint(p.data());
//
//			count++;
//
//			svo.insert(p);
//		}
//	}
//
//	vtkNew<vtkPolyData> polyData;
//	polyData->SetPoints(points);
//
//	WritePLY(polyData, "C:\\Debug\\GPV\\Original.ply");
//
//	//double spatialSigma = 0.5;  // adjust this based on the point cloud scale
//	//double featureSigma = 0.1;  // adjust based on feature variance
//	//double neighborhoodSize = 0.5;  // adjust based on the density of the point cloud
//
//	//// Apply bilateral filter
//	//vtkSmartPointer<vtkPoints> newPoints = BilateralFilter(polyData, spatialSigma, featureSigma, neighborhoodSize);
//
//	//vtkNew<vtkPolyData> newPolyData;
//	//newPolyData->SetPoints(newPoints);
//
//	//WritePLY(newPolyData, "C:\\Debug\\GPV\\Filtered.ply");
//
//	vtkNew<vtkVertexGlyphFilter> vertexFilter;
//	vertexFilter->SetInputData(polyData);
//	vertexFilter->Update();
//
//	vtkNew<vtkPolyDataMapper> mapper;
//	mapper->SetInputData(vertexFilter->GetOutput());
//
//	vtkNew<vtkActor> actor;
//	actor->SetMapper(mapper);
//
//	actor->GetProperty()->SetPointSize(5.0f);
//	actor->GetProperty()->SetColor(1.0, 0.0, 0.0);
//
//	renderer->AddActor(actor);
//
//	svo.visualize(renderer);
//}
//
//void Initialize()
//{
//	vector<Eigen::Vector3f> points;
//	for (size_t i = 0; i < size_0; i++)
//	{
//		auto& p = points_0[i];
//
//		if ((p.x() > -1000 && p.y() > -1000 && p.z() > -1000) &&
//			(p.x() < 1000 && p.y() < 1000 && p.z() < 1000))
//		{
//			points.push_back(p);
//		}
//	}
//
//	//Algorithm::TestSVO(points.data(), points.size());
//}
//
//int main() {
//    openvdb::initialize();
//
//    MaximizeConsoleWindowOnMonitor(1);
//
//    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
//    renderer->SetBackground(0.3, 0.5, 0.7);
//
//    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
//    renderWindow->SetSize(1920, 1080);
//	//renderWindow->SetSize(256, 480);
//    renderWindow->AddRenderer(renderer);
//
//    vtkSmartPointer<vtkRenderWindowInteractor> interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
//    //vtkNew<vtkInteractorStyleTrackballCamera> trackballStyle;
//    //interactor->SetInteractorStyle(trackballStyle);
//    vtkNew<CustomTrackballStyle> customTrackballStyle;
//    interactor->SetInteractorStyle(customTrackballStyle);
//    interactor->SetRenderWindow(renderWindow);
//    interactor->Initialize();
//
//    VisualDebugging::Initialize(renderer);
//
//    MaximizeVTKWindowOnMonitor(renderWindow, 2);
//
//	LoadPatch(3, renderer);
//
//	Initialize();
//
//	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { (float)128, 0, 0 }, 255, 0, 0);
//	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0, (float)240, 0 }, 0, 255, 0);
//	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0, 0, (float)240 }, 0, 0, 255);
//
//    vtkSmartPointer<vtkCallbackCommand> keyPressCallback = vtkSmartPointer<vtkCallbackCommand>::New();
//    keyPressCallback->SetCallback(OnKeyPress);
//    keyPressCallback->SetClientData(renderer);
//
//    vtkSmartPointer<TimerCallback> timerCallback = vtkSmartPointer<TimerCallback>::New();
//
//    interactor->AddObserver(vtkCommand::TimerEvent, timerCallback);
//    int timerId = interactor->CreateRepeatingTimer(16);
//    if (timerId < 0) {
//        std::cerr << "Error: Timer was not created!" << std::endl;
//    }
//
//    interactor->AddObserver(vtkCommand::KeyPressEvent, keyPressCallback);
//
//	//vtkCamera* camera = renderer->GetActiveCamera();
//	//camera->SetParallelProjection(true);
//	//renderer->ResetCamera();
//
//	CUDA::Test();
//
//    renderWindow->Render();
//    interactor->Start();
//
//    VisualDebugging::Terminate();
//
//    return 0;
//}
















#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkImageDataGeometryFilter.h>
#include <vtkWarpScalar.h>
#include <vtkMarchingCubes.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkNamedColors.h>
#include <vtkProperty.h>

int main() {
    // Step 1: Create height map as vtkImageData and extrude into depth to create a volume
    int width = 100;  // Width of height map
    int height = 100; // Height of height map
    int depth = 20;   // Number of layers in the Z direction to make it a volume

    vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
    imageData->SetDimensions(width, height, depth);
    imageData->AllocateScalars(VTK_FLOAT, 1);

    for (int z = 0; z < depth; z++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float* pixel = static_cast<float*>(imageData->GetScalarPointer(x, y, z));
                // Replace this with your actual height map values
                pixel[0] = static_cast<float>(x * y) / 100.0f; // Example height function
            }
        }
    }

    // Step 2: Apply Marching Cubes to extract an isosurface from the volume
    vtkSmartPointer<vtkMarchingCubes> marchingCubes = vtkSmartPointer<vtkMarchingCubes>::New();
    marchingCubes->SetInputData(imageData);
    marchingCubes->SetValue(0, 10.0); // Set the iso-value to extract. Adjust as needed.

    // Step 3: Create a mapper and actor for the Marching Cubes result
    vtkSmartPointer<vtkPolyDataMapper> marchingCubesMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    marchingCubesMapper->SetInputConnection(marchingCubes->GetOutputPort());

    vtkSmartPointer<vtkActor> marchingCubesActor = vtkSmartPointer<vtkActor>::New();
    marchingCubesActor->SetMapper(marchingCubesMapper);
    marchingCubesActor->GetProperty()->SetColor(1.0, 0.0, 0.0); // Set color of the isosurface (red)

    // Step 4: Convert the height map image data to geometry to visualize it
    vtkSmartPointer<vtkImageDataGeometryFilter> geometryFilter = vtkSmartPointer<vtkImageDataGeometryFilter>::New();
    geometryFilter->SetInputData(imageData);
    geometryFilter->Update();

    // Step 5: Warp the scalar values to create a 3D surface
    vtkSmartPointer<vtkWarpScalar> warpScalar = vtkSmartPointer<vtkWarpScalar>::New();
    warpScalar->SetInputConnection(geometryFilter->GetOutputPort());
    warpScalar->SetScaleFactor(1.0); // Adjust the scale factor to exaggerate or reduce height

    // Step 6: Create a mapper and actor for the height map surface
    vtkSmartPointer<vtkPolyDataMapper> heightMapMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    heightMapMapper->SetInputConnection(warpScalar->GetOutputPort());

    vtkSmartPointer<vtkActor> heightMapActor = vtkSmartPointer<vtkActor>::New();
    heightMapActor->SetMapper(heightMapMapper);
    heightMapActor->GetProperty()->SetColor(0.0, 1.0, 0.0); // Set color of the height map surface (green)
    heightMapActor->GetProperty()->SetOpacity(0.5); // Set opacity to see both surfaces simultaneously

    // Step 7: Set up rendering components
    vtkSmartPointer<vtkNamedColors> colors = vtkSmartPointer<vtkNamedColors>::New();

    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->AddActor(marchingCubesActor);
    renderer->AddActor(heightMapActor);
    renderer->SetBackground(colors->GetColor3d("SlateGray").GetData());

    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);
    renderWindow->SetSize(800, 600);

    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    // Step 8: Start the visualization
    renderWindow->Render();
    renderWindowInteractor->Start();

    return EXIT_SUCCESS;
}
