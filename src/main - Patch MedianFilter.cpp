#include <Common.h>
#include <App/App.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include <Algorithm/CustomPolyDataFilter.h>
#include <Algorithm/vtkMedianFilter.h>
#include <Algorithm/vtkQuantizingFilter.h>

#include <Debugging/VisualDebugging.h>

#include <CUDA/HashTable.cuh>
#include <CUDA/KDTree.cuh>
#include <CUDA/KDITree.cuh>
#include <CUDA/SVO.cuh>
#include <CUDA/AOctree.cuh>

#include <CUDA/Processing.cuh>

#include <CUDA/CUDA.cuh>

int depthIndex = 0;

namespace SVO
{
	struct Triangle
	{
		Eigen::Vector3f vertices[3];
	};

	struct NearestNeighborResult {
		Eigen::Vector3f nearestPoint;
		float distanceSquared = std::numeric_limits<float>::max();
	};

	// Define the edgeConnection table
	const int edgeConnection[12][2] = {
		{0, 1}, {1, 3}, {3, 2}, {2, 0}, // Edges along z = 0 face
		{4, 5}, {5, 7}, {7, 6}, {6, 4}, // Edges along z = 1 face
		{0, 4}, {1, 5}, {2, 6}, {3, 7}  // Edges between z = 0 and z = 1 faces
	};

	// Define the corners array for a voxel's corner positions based on node center and halfSize
	std::vector<Eigen::Vector3f> GetVoxelCorners(const Eigen::Vector3f& center, float halfSize)
	{
		return {
			center + Eigen::Vector3f(-halfSize, -halfSize, -halfSize),
			center + Eigen::Vector3f(halfSize, -halfSize, -halfSize),
			center + Eigen::Vector3f(-halfSize,  halfSize, -halfSize),
			center + Eigen::Vector3f(halfSize,  halfSize, -halfSize),
			center + Eigen::Vector3f(-halfSize, -halfSize,  halfSize),
			center + Eigen::Vector3f(halfSize, -halfSize,  halfSize),
			center + Eigen::Vector3f(-halfSize,  halfSize,  halfSize),
			center + Eigen::Vector3f(halfSize,  halfSize,  halfSize)
		};
	}

	struct SVONode
	{
		bool isLeaf = false;
		size_t parentIndex = Max.U64;
		size_t childIndex[8] = { Max.U64, Max.U64, Max.U64, Max.U64, Max.U64, Max.U64, Max.U64, Max.U64 };
		size_t voxelIndex = Max.U64;

		Eigen::Vector3f center = Eigen::Vector3f(0.0f, 0.0f, 0.0f);  // Center of the voxel
		float size = 0.0f;  // Size of the voxel edge length

		float sdfValue = Max.FLOAT;  // Signed Distance Function value
		bool occupied = false;       // Indicates if this voxel is occupied
		int updatedCount = 0;
	};

	void InitializeNode(SVONode* node, const Eigen::Vector3f& center, float size)
	{
		node->isLeaf = false;
		node->parentIndex = Max.U64;
		node->voxelIndex = Max.U64;
		node->childIndex[0] = Max.U64;
		node->childIndex[1] = Max.U64;
		node->childIndex[2] = Max.U64;
		node->childIndex[3] = Max.U64;
		node->childIndex[4] = Max.U64;
		node->childIndex[5] = Max.U64;
		node->childIndex[6] = Max.U64;
		node->childIndex[7] = Max.U64;
		node->center = center;
		node->size = size;
		node->sdfValue = Max.FLOAT;
		node->occupied = false;
		node->updatedCount = 0;
	}

	struct Octree
	{
		SVONode* nodes = nullptr;
		size_t numberOfNodes = 0;

		size_t nodeBufferIndex = 0;
		size_t rootIndex = Max.U64;
		size_t maxDepth = 0;
	};

	int GetChildIndex(const Eigen::Vector3f& point, const Eigen::Vector3f& nodeCenter);
	void UpdateNodeWithSDF(SVONode& node, const Eigen::Vector3f& point);
	float ComputeSDFValue(const SVONode& node, const Eigen::Vector3f& point);

	void InitializeSVO(Octree* octree, size_t numberOfNodes)
	{
		octree->nodes = new SVONode[numberOfNodes];
		octree->numberOfNodes = numberOfNodes;
		octree->nodeBufferIndex = 0;
		octree->rootIndex = Max.U64;
		octree->maxDepth = 0;
	}

	void IntegratePointCloud(Octree* octree, const Eigen::Vector3f* pointCloud, size_t numberOfPoints, float rootVoxelSize, int maxDepth)
	{
		for (size_t i = 0; i < numberOfPoints; ++i)
		{
			const Eigen::Vector3f& point = pointCloud[i];

			size_t currentIndex = octree->rootIndex;

			if (currentIndex == Max.U64)
			{
				// Create a root node if it doesn't exist
				octree->rootIndex = 0;
				InitializeNode(&octree->nodes[octree->rootIndex], Eigen::Vector3f(0.0f, 0.0f, 0.0f), rootVoxelSize);
				currentIndex = octree->rootIndex;
			}

			int currentDepth = 0;

			while (!octree->nodes[currentIndex].isLeaf)
			{
				if (currentDepth >= maxDepth || /* another condition, such as point threshold */ false)
				{
					octree->nodes[currentIndex].isLeaf = true; // Stop subdividing
					break;
				}

				Eigen::Vector3f nodeCenter = octree->nodes[currentIndex].center;
				float currentVoxelSize = octree->nodes[currentIndex].size;

				int childIdx = GetChildIndex(point, nodeCenter);

				if (octree->nodes[currentIndex].childIndex[childIdx] == Max.U64)
				{
					octree->nodes[currentIndex].childIndex[childIdx] = ++octree->nodeBufferIndex;

					float halfSize = currentVoxelSize / 2.0f;
					Eigen::Vector3f childCenter = nodeCenter;

					childCenter.x() += (childIdx & 1) ? halfSize / 2.0f : -halfSize / 2.0f;
					childCenter.y() += (childIdx & 2) ? halfSize / 2.0f : -halfSize / 2.0f;
					childCenter.z() += (childIdx & 4) ? halfSize / 2.0f : -halfSize / 2.0f;

					//printf("octree->nodeBufferIndex : %llu, halfSize : %f\n", octree->nodeBufferIndex, halfSize);

					InitializeNode(&octree->nodes[octree->nodeBufferIndex], childCenter, halfSize);
				}

				currentIndex = octree->nodes[currentIndex].childIndex[childIdx];
				currentDepth++;

				if (currentDepth > octree->maxDepth)
				{
					octree->maxDepth = currentDepth;
					printf("octree->maxDepth : %d\n", octree->maxDepth);
				}
			}

			// Update leaf node with point information
			UpdateNodeWithSDF(octree->nodes[currentIndex], point);
		}
	}

	int GetChildIndex(const Eigen::Vector3f& point, const Eigen::Vector3f& nodeCenter)
	{
		int childIndex = 0;

		// Determine which half along each axis the point lies in
		if (point.x() >= nodeCenter.x()) childIndex |= 1;  // Set bit 0
		if (point.y() >= nodeCenter.y()) childIndex |= 2;  // Set bit 1
		if (point.z() >= nodeCenter.z()) childIndex |= 4;  // Set bit 2

		// The resulting `childIndex` is in the range [0, 7], representing one of the eight child nodes.
		return childIndex;
	}

	void UpdateNodeWithSDF(SVONode& node, const Eigen::Vector3f& point)
	{
		float sdfValue = ComputeSDFValue(node, point);
		//if (sdfValue != 0.0f)
		//{
		//	printf("sdfValue : %f\n", sdfValue);
		//}

		// If updating with multiple points, consider an averaging approach
		if (node.updatedCount > 0) {
			node.sdfValue = (node.sdfValue * node.updatedCount + sdfValue) / (node.updatedCount + 1);
		}
		else {
			node.sdfValue = sdfValue;
		}
		node.updatedCount++;  // Track the number of updates to the node

		// Update occupancy status based on SDF
		node.occupied = (node.sdfValue < 0.0f);  // Set based on average SDF

		//if (node.occupied)
		//{
		//	printf("OK\n");
		//}
	}

	float ComputeSDFValue(const SVONode& node, const Eigen::Vector3f& point)
	{
		Eigen::Vector3f toPoint = point - node.center;
		float halfSize = node.size / 2.0f;

		//float dx = std::max(0.0f, std::abs(toPoint.x()) - halfSize);
		//float dy = std::max(0.0f, std::abs(toPoint.y()) - halfSize);
		//float dz = std::max(0.0f, std::abs(toPoint.z()) - halfSize);

		float dx = std::abs(toPoint.x()) - halfSize;
		float dy = std::abs(toPoint.y()) - halfSize;
		float dz = std::abs(toPoint.z()) - halfSize;

		// Debugging output
		//std::cout << "dx: " << dx << ", dy: " << dy << ", dz: " << dz << std::endl;

		float sdfValue = std::sqrt(dx * dx + dy * dy + dz * dz);

		if (std::abs(toPoint.x()) < halfSize &&
			std::abs(toPoint.y()) < halfSize &&
			std::abs(toPoint.z()) < halfSize)
		{
			sdfValue = -sdfValue;
		}

		return sdfValue;
	}

	//float ComputeSDFValue(const SVONode& node, const Eigen::Vector3f& point)
	//{
	//	// Calculate the vector from the node's center to the point
	//	Eigen::Vector3f toPoint = point - node.center;

	//	// Calculate the half-size of the node (distance from the center to any face)
	//	float halfSize = node.size / 2.0f;

	//	// Distance to the closest surface of the cubic node
	//	float dx = std::max(0.0f, std::abs(toPoint.x()) - halfSize);
	//	float dy = std::max(0.0f, std::abs(toPoint.y()) - halfSize);
	//	float dz = std::max(0.0f, std::abs(toPoint.z()) - halfSize);

	//	// SDF value is the Euclidean distance from point to nearest node surface
	//	float sdfValue = std::sqrt(dx * dx + dy * dy + dz * dz);

	//	// Heuristic: Negative if point is inside the node
	//	if (std::abs(toPoint.x()) < halfSize &&
	//		std::abs(toPoint.y()) < halfSize &&
	//		std::abs(toPoint.z()) < halfSize)
	//	{
	//		sdfValue = -sdfValue;  // Inside the node
	//	}

	//	return sdfValue;
	//}

	// Modified traversal function with depth info
	void TraverseOctree(Octree* octree, size_t nodeIndex, int currentDepth, const std::function<void(const SVONode&, int)>& operation)
	{
		if (nodeIndex == Max.U64)
		{
			return; // No node exists at this index
		}

		const SVONode& currentNode = octree->nodes[nodeIndex];

		// Perform the operation on the current node, passing the depth information
		operation(currentNode, currentDepth);

		// If the current node is a leaf, no need to traverse further
		if (currentNode.isLeaf)
		{
			return;
		}

		// Recursively traverse all the children, incrementing the depth
		for (int i = 0; i < 8; ++i)
		{
			if (currentNode.childIndex[i] != Max.U64)
			{
				TraverseOctree(octree, currentNode.childIndex[i], currentDepth + 1, operation);
			}
		}
	}

	// Function to start traversal from the root node
	void TraverseOctree(Octree* octree, const std::function<void(const SVONode&, int)>& operation)
	{
		if (octree->rootIndex != Max.U64)
		{
			TraverseOctree(octree, octree->rootIndex, 0, operation); // Start traversal with depth = 0
		}
		else
		{
			std::cout << "The octree is empty." << std::endl;
		}
	}

	void NearestNeighborDFS(Octree* octree, size_t nodeIndex, const Eigen::Vector3f& queryPoint, NearestNeighborResult& result) {
		if (nodeIndex == Max.U64) {
			return; // Invalid node index, terminate this branch.
		}

		// Get the current node from the octree.
		const SVONode& currentNode = octree->nodes[nodeIndex];

		// Calculate distance from query point to the node center.
		float distSquaredToCenter = (queryPoint - currentNode.center).squaredNorm();

		// Prune nodes that cannot possibly provide a closer point.
		if (distSquaredToCenter > result.distanceSquared) {
			return;
		}

		// If it's a leaf node and is occupied, check if it's closer than the current best.
		if (currentNode.isLeaf && currentNode.occupied) {
			if (distSquaredToCenter < result.distanceSquared) {
				result.distanceSquared = distSquaredToCenter;
				result.nearestPoint = currentNode.center;
			}
			return; // No need to explore further since it's a leaf.
		}

		// Create a list of child indices and their distances to the query point.
		std::vector<std::pair<size_t, float>> childrenDistances;
		for (int i = 0; i < 8; ++i) {
			if (currentNode.childIndex[i] != Max.U64) {
				const SVONode& childNode = octree->nodes[currentNode.childIndex[i]];
				float childDistSquared = (queryPoint - childNode.center).squaredNorm();
				childrenDistances.emplace_back(currentNode.childIndex[i], childDistSquared);
			}
		}

		// Sort the children by their distances to the query point.
		std::sort(childrenDistances.begin(), childrenDistances.end(),
			[](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) {
				return a.second < b.second;
			});

		// Recursively visit each child in the order of proximity.
		for (const auto& child : childrenDistances) {
			NearestNeighborDFS(octree, child.first, queryPoint, result);
		}
	}

	// Utility to get SDF values at the eight corners of a voxel
	void GetCornerSDFValues(const SVONode& node, std::array<float, 8>& cornerSDFs)
	{
		float halfSize = node.size / 2.0f;
		const Eigen::Vector3f center = node.center;

		std::vector<Eigen::Vector3f> corners = {
			center + Eigen::Vector3f(-halfSize, -halfSize, -halfSize),
			center + Eigen::Vector3f(halfSize, -halfSize, -halfSize),
			center + Eigen::Vector3f(-halfSize,  halfSize, -halfSize),
			center + Eigen::Vector3f(halfSize,  halfSize, -halfSize),
			center + Eigen::Vector3f(-halfSize, -halfSize,  halfSize),
			center + Eigen::Vector3f(halfSize, -halfSize,  halfSize),
			center + Eigen::Vector3f(-halfSize,  halfSize,  halfSize),
			center + Eigen::Vector3f(halfSize,  halfSize,  halfSize)
		};

		// Compute SDF values for each corner
		for (int i = 0; i < 8; ++i)
		{
			cornerSDFs[i] = ComputeSDFValue(node, corners[i]);
		}
	}

	// Marching Cubes table (only partial here; full table needed in practice)
	const int edgeTable[256] = { /* ... */ };
	const int triTable[256][16] = { /* ... */ };

	// Perform Marching Cubes for a single voxel
	void MarchingCubes(const SVONode& node, std::vector<Triangle>& triangles)
	{
		std::array<float, 8> cornerSDFs;
		GetCornerSDFValues(node, cornerSDFs);

		int cubeIndex = 0;
		for (int i = 0; i < 8; ++i)
		{
			if (cornerSDFs[i] < 0.0f) // Inside the surface
			{
				cubeIndex |= (1 << i);
			}
		}

		if (edgeTable[cubeIndex] == 0)
		{
			return;
		}

		// Retrieve voxel corner positions
		float halfSize = node.size / 2.0f;
		const auto corners = GetVoxelCorners(node.center, halfSize);

		Eigen::Vector3f vertList[12];

		for (int i = 0; i < 12; ++i)
		{
			if (edgeTable[cubeIndex] & (1 << i))
			{
				int idx1 = edgeConnection[i][0];
				int idx2 = edgeConnection[i][1];

				const Eigen::Vector3f& p1 = corners[idx1];
				const Eigen::Vector3f& p2 = corners[idx2];
				float val1 = cornerSDFs[idx1];
				float val2 = cornerSDFs[idx2];

				float t = val1 / (val1 - val2);
				vertList[i] = p1 + t * (p2 - p1);
			}
		}

		for (int i = 0; triTable[cubeIndex][i] != -1; i += 3)
		{
			Triangle tri;
			tri.vertices[0] = vertList[triTable[cubeIndex][i]];
			tri.vertices[1] = vertList[triTable[cubeIndex][i + 1]];
			tri.vertices[2] = vertList[triTable[cubeIndex][i + 2]];
			triangles.push_back(tri);

			VisualDebugging::AddTriangle("Triangles", tri.vertices[0], tri.vertices[1], tri.vertices[2], Color4::White);
		}
	}

	// Main function to extract triangles from the octree
	void ExtractTrianglesFromOctree(Octree* octree, std::vector<Triangle>& triangles)
	{
		TraverseOctree(octree, [&](const SVONode& node, int depth)
			{
				if (node.isLeaf && node.occupied)
				{
					// Only process leaf nodes that are occupied (surface boundaries)
					MarchingCubes(node, triangles);
				}
			});
	}
}


int pid = 0;
size_t size_0 = 0;
size_t size_45 = 0;
float transform_0[16];
float transform_45[16];
unsigned char image_0[400 * 480];
unsigned char image_45[400 * 480];
Eigen::Vector3f points_0[400 * 480];
Eigen::Vector3f points_45[400 * 480];
Eigen::AlignedBox3f aabb(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));
Eigen::AlignedBox3f taabb(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));
Eigen::AlignedBox3f lmax(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));

SVO::Octree* pOctree = nullptr;

vector<Eigen::Vector3f> patchPoints;
vector<Eigen::Vector3f> inputPoints;

void LoadPatch(int patchID, vtkRenderer* renderer)
{
	stringstream ss;
	ss << "C:\\Debug\\Patches\\patch_" << patchID << ".pat";

	ifstream ifs;
	ifs.open(ss.str(), ios::in | ios::binary);

	ifs.read((char*)&pid, sizeof(int));
	ifs.read((char*)&size_0, sizeof(size_t));
	ifs.read((char*)&size_45, sizeof(size_t));
	ifs.read((char*)&transform_0, sizeof(float) * 16);
	ifs.read((char*)&transform_45, sizeof(float) * 16);
	ifs.read((char*)&image_0, sizeof(unsigned char) * 400 * 480);
	ifs.read((char*)&image_45, sizeof(unsigned char) * 400 * 480);
	ifs.read((char*)&points_0, sizeof(Eigen::Vector3f) * size_0);
	ifs.read((char*)&points_45, sizeof(Eigen::Vector3f) * size_45);

	ifs.close();

	vtkNew<vtkPoints> points;
	//points->SetNumberOfPoints(size_0);

	Eigen::Matrix4f t0(transform_0);
	Eigen::Matrix4f t45(transform_45);

	aabb = Eigen::AlignedBox3f(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));

	int count = 0;
	
	patchPoints.clear();

	for (size_t i = 0; i < size_0; i++)
	{
		auto& p = points_0[i];

		if ((p.x() > -1000 && p.y() > -1000 && p.z() > -1000) &&
			(p.x() < 1000 && p.y() < 1000 && p.z() < 1000))
		{
			Eigen::Vector4f p4(p.x(), p.y(), p.z(), 1.0f);
			Eigen::Vector4f tp = t0 * p4;
			Eigen::Vector3f tp3 = Eigen::Vector3f(tp.x(), tp.y(), tp.z());

			points->InsertNextPoint(tp3.data());

			aabb.extend(tp3);
			taabb.extend(tp3);

			lmax.extend(aabb.max() - aabb.min());

			count++;

			patchPoints.push_back(tp3);
		}
	}

	for (size_t i = 0; i < size_45; i++)
	{
		auto& p = points_45[i];

		if ((p.x() > -1000 && p.y() > -1000 && p.z() > -1000) &&
			(p.x() < 1000 && p.y() < 1000 && p.z() < 1000))
		{
			Eigen::Vector4f p4(p.x(), p.y(), p.z(), 1.0f);
			Eigen::Vector4f tp = t45 * p4;
			Eigen::Vector3f tp3 = Eigen::Vector3f(tp.x(), tp.y(), tp.z());

			points->InsertNextPoint(tp3.data());

			aabb.extend(tp3);
			taabb.extend(tp3);

			lmax.extend(aabb.max() - aabb.min());

			count++;

			patchPoints.push_back(tp3);
		}
	}

	inputPoints.insert(inputPoints.end(), patchPoints.begin(), patchPoints.end());

	std::cout << aabb.min() << std::endl;
	std::cout << aabb.max() << std::endl;

	return;

	//VisualDebugging::AddCube("aabb", (aabb.min() + aabb.max()) * 0.5f, aabb.max() - aabb.min(), { 0.0f, 0.0f, 0.0f }, Color4::Red);

	vtkNew<vtkPolyData> polyData;
	polyData->SetPoints(points);

	//WritePLY(polyData, "C:\\Debug\\GPV\\Original.ply");

	//double spatialSigma = 0.5;  // adjust this based on the point cloud scale
	//double featureSigma = 0.1;  // adjust based on feature variance
	//double neighborhoodSize = 0.5;  // adjust based on the density of the point cloud

	//// Apply bilateral filter
	//vtkSmartPointer<vtkPoints> newPoints = BilateralFilter(polyData, spatialSigma, featureSigma, neighborhoodSize);

	//vtkNew<vtkPolyData> newPolyData;
	//newPolyData->SetPoints(newPoints);

	//WritePLY(newPolyData, "C:\\Debug\\GPV\\Filtered.ply");

	vtkNew<vtkVertexGlyphFilter> vertexFilter;
	vertexFilter->SetInputData(polyData);
	vertexFilter->Update();

	vtkNew<vtkPolyDataMapper> mapper;
	mapper->SetInputData(vertexFilter->GetOutput());

	vtkNew<vtkActor> actor;
	actor->SetObjectName("points");
	actor->SetMapper(mapper);

	actor->GetProperty()->SetPointSize(5.0f);
	actor->GetProperty()->SetColor(1.0, 1.0, 1.0);

	renderer->AddActor(actor);
}

void OnKeyPress(vtkObject* caller, long unsigned int eventId, void* clientData, void* callData) {
	vtkRenderWindowInteractor* interactor = static_cast<vtkRenderWindowInteractor*>(caller);
	vtkRenderWindow* renderWindow = interactor->GetRenderWindow();
	vtkRenderer* renderer = renderWindow->GetRenderers()->GetFirstRenderer();

	std::string key = interactor->GetKeySym();

	printf("%s\n", key.c_str());

	if (key == "r")
	{
		std::cout << "Key 'r' was pressed. Resetting camera." << std::endl;
		vtkRenderer* renderer = static_cast<vtkRenderer*>(clientData);

		vtkCamera* camera = renderer->GetActiveCamera();
		renderer->ResetCamera();

		// Reset the camera position, focal point, and view up vector to their defaults
		camera->SetPosition(0, 0, 1);          // Reset position to default
		camera->SetFocalPoint(0, 0, 0);        // Reset focal point to origin
		camera->SetViewUp(0, 1, 0);            // Reset the up vector to default (Y-axis up)

		interactor->Render();
	}
	else if (key == "c")
	{
		vtkRenderer* renderer = static_cast<vtkRenderer*>(clientData);

		vtkCamera* camera = renderer->GetActiveCamera();
		camera->SetParallelProjection(!camera->GetParallelProjection());

		interactor->Render();
	}
	else if (key == "Escape")
	{
		std::cout << "Key 'Escape' was pressed. Exiting." << std::endl;
		interactor->TerminateApp();
	}
	else if (key == "minus")
	{
		VisualDebugging::SetLineWidth("Spheres", 1);
		vtkSmartPointer<vtkActor> actor = VisualDebugging::GetSphereActor("Spheres");
		vtkSmartPointer<vtkMapper> mapper = actor->GetMapper();
		vtkSmartPointer<vtkPolyDataMapper> polyDataMapper =
			vtkPolyDataMapper::SafeDownCast(mapper);
		vtkSmartPointer<vtkGlyph3DMapper> glyph3DMapper = vtkGlyph3DMapper::SafeDownCast(mapper);
		vtkSmartPointer<vtkPolyData> polyData = vtkPolyData::SafeDownCast(glyph3DMapper->GetInputDataObject(0, 0));
		vtkSmartPointer<vtkPointData> pointData = polyData->GetPointData();
		vtkSmartPointer<vtkDoubleArray> scaleArray =
			vtkDoubleArray::SafeDownCast(pointData->GetArray("Scales"));
		for (vtkIdType i = 0; i < scaleArray->GetNumberOfTuples(); ++i)
		{
			double scale[3]; // Assuming 3-component scale array (X, Y, Z)
			scaleArray->GetTuple(i, scale);
			//std::cout << "Scale for point " << i << ": "
			//	<< scale[0 ] << ", " << scale[1] << ", " << scale[2] << std::endl;
			scale[0] *= 0.9;
			scale[1] *= 0.9;
			scale[2] *= 0.9;
			scaleArray->SetTuple(i, scale);
		}
		polyData->Modified();
		glyph3DMapper->SetScaleArray("Scales");
		glyph3DMapper->Update();
	}
	else if (key == "equal")
	{
		VisualDebugging::SetLineWidth("Spheres", 1);
		vtkSmartPointer<vtkActor> actor = VisualDebugging::GetSphereActor("Spheres");
		vtkSmartPointer<vtkMapper> mapper = actor->GetMapper();
		vtkSmartPointer<vtkPolyDataMapper> polyDataMapper =
			vtkPolyDataMapper::SafeDownCast(mapper);
		vtkSmartPointer<vtkGlyph3DMapper> glyph3DMapper = vtkGlyph3DMapper::SafeDownCast(mapper);
		vtkSmartPointer<vtkPolyData> polyData = vtkPolyData::SafeDownCast(glyph3DMapper->GetInputDataObject(0, 0));
		vtkSmartPointer<vtkPointData> pointData = polyData->GetPointData();
		vtkSmartPointer<vtkDoubleArray> scaleArray =
			vtkDoubleArray::SafeDownCast(pointData->GetArray("Scales"));
		for (vtkIdType i = 0; i < scaleArray->GetNumberOfTuples(); ++i)
		{
			double scale[3]; // Assuming 3-component scale array (X, Y, Z)
			scaleArray->GetTuple(i, scale);
			//std::cout << "Scale for point " << i << ": "
			//	<< scale[0 ] << ", " << scale[1] << ", " << scale[2] << std::endl;
			scale[0] *= 1.1;
			scale[1] *= 1.1;
			scale[2] *= 1.1;
			scaleArray->SetTuple(i, scale);
		}
		polyData->Modified();
		glyph3DMapper->SetScaleArray("Scales");
		glyph3DMapper->Update();
	}
	else if (key == "Return")
	{
		vtkRenderer* renderer = static_cast<vtkRenderer*>(clientData);
		vtkRenderWindow* renderWindow = renderer->GetRenderWindow();

		// Grab the depth buffer (z-values)
		int width = renderWindow->GetSize()[0];
		int height = renderWindow->GetSize()[1];
		vtkSmartPointer<vtkFloatArray> depthBuffer = vtkSmartPointer<vtkFloatArray>::New();
		depthBuffer->SetNumberOfComponents(1);
		depthBuffer->SetNumberOfTuples(width * height);

		renderWindow->GetZbufferData(0, 0, width - 1, height - 1, depthBuffer);

		// Save depth map to an image
		float minDepth = std::numeric_limits<float>::max();
		float maxDepth = std::numeric_limits<float>::lowest();

		for (vtkIdType i = 0; i < depthBuffer->GetNumberOfTuples(); i++) {
			float depthValue = depthBuffer->GetValue(i);
			minDepth = std::min(minDepth, depthValue);
			maxDepth = std::max(maxDepth, depthValue);
		}

		vtkSmartPointer<vtkImageData> depthImage = vtkSmartPointer<vtkImageData>::New();
		depthImage->SetDimensions(width, height, 1);
		depthImage->AllocateScalars(VTK_FLOAT, 1);

		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				float depthValue = depthBuffer->GetValue((height - y - 1) * width + x);
				float normalizedDepth = (depthValue - minDepth) / (maxDepth - minDepth);

				depthImage->SetScalarComponentFromFloat(x, y, 0, 0, normalizedDepth * 255);

				if (0.0f != normalizedDepth)
				{
					VisualDebugging::AddSphere("Depth", { (float)x / (float)width * 100.0f, (float)y / (float)height * 100.0f, normalizedDepth * 100.0f }, { 0.1f, 0.1f, 0.1f }, { 0.0f, 0.0f, 0.0f }, Color4::White);
				}
			}
		}

		// Optional: Cast the float image to unsigned char to save as PNG
		vtkSmartPointer<vtkImageCast> castFilter = vtkSmartPointer<vtkImageCast>::New();
		castFilter->SetInputData(depthImage);
		castFilter->SetOutputScalarTypeToUnsignedChar();
		castFilter->Update();

		vtkSmartPointer<vtkPNGWriter> writer = vtkSmartPointer<vtkPNGWriter>::New();
		writer->SetFileName("c:\\Debug\\2D\\depthmap.png");
		writer->SetInputData(castFilter->GetOutput());
		writer->Write();
	}
	else if (key == "1")
	{
		VisualDebugging::ToggleVisibility("Original");
	}
	else if (key == "2")
	{
		VisualDebugging::ToggleVisibility("DownSample");
	}
	else if (key == "3")
	{
		VisualDebugging::ToggleVisibility("Filter");
	}
	else if (key == "4")
	{
		VisualDebugging::ToggleVisibility("WiredBox_3");
	}
	else if (key == "5")
	{
		VisualDebugging::ToggleVisibility("WiredBox_4");
	}
	else if (key == "6")
	{
		VisualDebugging::ToggleVisibility("WiredBox_5");
	}
	else if (key == "7")
	{
		VisualDebugging::ToggleVisibility("WiredBox_6");
	}
	else if (key == "8")
	{
		VisualDebugging::ToggleVisibility("WiredBox_7");
	}
	else if (key == "9")
	{
		VisualDebugging::ToggleVisibility("WiredBox_8");
	}
	else if (key == "0")
	{
		VisualDebugging::ToggleVisibility("WiredBox_9");
	}
	else if (key == "Left")
	{
		depthIndex--;
		if (depthIndex < 0) depthIndex = 0;

		for (int i = 0; i < pOctree->maxDepth; i++)
		{
			stringstream ss;
			ss << "WiredBox_" << i;
			VisualDebugging::SetVisibility(ss.str(), false);
		}
		{
			stringstream ss;
			ss << "WiredBox_" << depthIndex;
			VisualDebugging::SetVisibility(ss.str(), true);
		}
	}
	else if (key == "Right")
	{
		depthIndex++;
		if (depthIndex > pOctree->maxDepth) depthIndex = pOctree->maxDepth;

		for (int i = 0; i < pOctree->maxDepth; i++)
		{
			stringstream ss;
			ss << "WiredBox_" << i;
			VisualDebugging::SetVisibility(ss.str(), false);
		}
		{
			stringstream ss;
			ss << "WiredBox_" << depthIndex;
			VisualDebugging::SetVisibility(ss.str(), true);
		}
	}
	else if (key == "space")
	{
	}
}

vector<Eigen::Vector3f> GaussianBlur(const vector<Eigen::Vector3f>& pointCloud, double radius, double sigma) {
	vector<Eigen::Vector3f> blurredPointCloud(pointCloud.size());

	// Parameters
	double sigmaSq = sigma * sigma;

	// For each point in the point cloud
	for (size_t i = 0; i < pointCloud.size(); ++i) {
		Eigen::Vector3f newPos = Eigen::Vector3f::Zero();
		double weightSum = 0.0;

		for (size_t j = 0; j < pointCloud.size(); ++j) {
			if (i == j) continue;  // Skip itself

			// Compute distance between points i and j
			double distance = (pointCloud[i] - pointCloud[j]).norm();

			// If point j is within the specified radius
			if (distance < radius) {
				// Compute Gaussian weight
				double weight = std::exp(-(distance * distance) / (2 * sigmaSq));
				weightSum += weight;

				// Apply weight to the neighbor point's position
				newPos += weight * pointCloud[j];
			}
		}

		// Normalize to get the blurred position
		if (weightSum > 0) {
			newPos /= weightSum;
		}
		else {
			newPos = pointCloud[i];  // No neighbors, keep original position
		}

		// Assign new blurred position to the result
		blurredPointCloud[i] = newPos;
	}

	return blurredPointCloud;
}

void TestManaged(vtkRenderer* renderer)
{
	auto tct = Time::Now();
	auto grid = CUDATest();
	Time::End(tct, "CUDATest()");

	auto ta = Time::Now();
	vector<Eigen::Vector3f> dpoints;
	for (size_t i = 0; i < 500 * 500 * 500; i++)
	{
		auto voxel = grid[i];

		if (voxel.value < 1.0f)
		{
			dpoints.push_back(Eigen::Vector3f(voxel.x, voxel.y, voxel.z));
		}
	}
	Time::End(ta, "Access");

	vtkNew<vtkPoints> points;

	for (size_t i = 0; i < dpoints.size(); i++)
	{
		auto& p = dpoints[i];
		points->InsertNextPoint(p.x(), p.y(), p.z());
		//VisualDebugging::AddSphere("Spheres", { p.x(), p.y(), p.z() }, { 0.01f, 0.01f, 0.01f }, { 0.0f, 0.0f, 0.0f }, Color4::White);
	}

	vtkNew<vtkPolyData> polyData;
	polyData->SetPoints(points);

	vtkNew<vtkVertexGlyphFilter> vertexFilter;
	vertexFilter->SetInputData(polyData);
	vertexFilter->Update();

	vtkNew<vtkPolyDataMapper> mapper;
	mapper->SetInputData(vertexFilter->GetOutput());

	vtkNew<vtkActor> actor;
	actor->SetMapper(mapper);

	actor->GetProperty()->SetPointSize(0.05f);
	actor->GetProperty()->SetColor(1.0, 1.0, 1.0);

	renderer->AddActor(actor);

	cudaFree(grid);
}

struct kdiNode {
	int index; // points 배열에서의 인덱스를 저장
	kdiNode* left;
	kdiNode* right;
};

class kdiTree {
public:
	kdiTree(Eigen::Vector3f* points, int size) : points(points), size(size) {
		std::vector<int> indices(size);
		for (int i = 0; i < size; ++i) {
			indices[i] = i;
		}
		root = buildTree(indices.data(), indices.data() + size, 0);
	}

	~kdiTree() {
		destroyTree(root);
	}

	// 범위 검색 함수 추가
	void rangeSearch(const Eigen::Vector3f& min, const Eigen::Vector3f& max, std::vector<int>& results) const {
		rangeSearchRec(root, min, max, 0, results);
	}

	// 반경 검색 함수 추가
	void radiusSearch(const Eigen::Vector3f& center, float radius, std::vector<int>& results) const {
		radiusSearchRec(root, center, radius, 0, results);
	}

	// 최근접 이웃 검색 함수 추가
	int nearestNeighbor(const Eigen::Vector3f& target) const {
		float bestDist = std::numeric_limits<float>::max();
		int bestIndex = -1;
		nearestNeighborRec(root, target, 0, bestDist, bestIndex);
		return bestIndex;
	}

	// k-최근접 이웃 검색 함수 추가
	void kNearestNeighbors(const Eigen::Vector3f& target, int k, std::vector<int>& results) const {
		std::vector<std::pair<float, int>> neighbors;
		kNearestNeighborsRec(root, target, 0, k, neighbors);

		for (const auto& neighbor : neighbors)
		{
			results.push_back(neighbor.second);
		}
	}

	// k-최근접 이웃 중 반경 내 검색 함수 추가
	void kNearestNeighborsWithinRadius(const Eigen::Vector3f& target, int k, float radius, std::vector<int>& results) const {
		std::vector<std::pair<float, int>> neighbors;
		kNearestNeighborsRec(root, target, 0, k, neighbors);

		// 반경 내의 이웃들만 결과에 추가
		for (const auto& neighbor : neighbors) {
			if (std::sqrt(neighbor.first) <= radius) {
				results.push_back(neighbor.second);
			}
		}
	}

	// k-최근접 이웃 중 범위 내 검색 함수 추가
	void kNearestNeighborsWithinRange(const Eigen::Vector3f& target, int k, const Eigen::Vector3f& min, const Eigen::Vector3f& max, std::vector<int>& results) const {
		std::vector<std::pair<float, int>> neighbors;
		kNearestNeighborsRec(root, target, 0, k, neighbors);

		// 범위 내의 이웃들만 결과에 추가
		for (const auto& neighbor : neighbors) {
			const Eigen::Vector3f& point = points[neighbor.second];
			bool inRange = true;
			for (int i = 0; i < 3; ++i) {
				if (point[i] < min[i] || point[i] > max[i]) {
					inRange = false;
					break;
				}
			}
			if (inRange) {
				results.push_back(neighbor.second);
			}
		}
	}

private:
	kdiNode* root;
	Eigen::Vector3f* points;
	int size;

	kdiNode* buildTree(int* start, int* end, int depth) {
		if (start >= end) {
			return nullptr;
		}

		int axis = depth % 3; // 3차원 공간이므로 0, 1, 2 축 반복
		int* mid = start + (end - start) / 2;

		// 중간값을 찾기 위해 정렬 수행
		std::nth_element(start, mid, end, [this, axis](int lhs, int rhs) {
			return points[lhs][axis] < points[rhs][axis];
			});

		// 새로운 노드 생성 및 초기화
		kdiNode* node = new kdiNode;
		node->index = *mid;
		node->left = buildTree(start, mid, depth + 1);  // 왼쪽 서브트리 재귀 호출
		node->right = buildTree(mid + 1, end, depth + 1); // 오른쪽 서브트리 재귀 호출

		return node;
	}

	void destroyTree(kdiNode* node) {
		if (node != nullptr) {
			destroyTree(node->left);
			destroyTree(node->right);
			delete node;
		}
	}

	void rangeSearchRec(kdiNode* node, const Eigen::Vector3f& min, const Eigen::Vector3f& max, int depth, std::vector<int>& results) const {
		if (node == nullptr) {
			return;
		}

		const Eigen::Vector3f& point = points[node->index];
		bool inRange = true;
		for (int i = 0; i < 3; ++i) {
			if (point[i] < min[i] || point[i] > max[i]) {
				inRange = false;
				break;
			}
		}

		if (inRange) {
			results.push_back(node->index);
		}

		int axis = depth % 3;
		if (point[axis] >= min[axis]) {
			rangeSearchRec(node->left, min, max, depth + 1, results);
		}
		if (point[axis] <= max[axis]) {
			rangeSearchRec(node->right, min, max, depth + 1, results);
		}
	}

	void radiusSearchRec(kdiNode* node, const Eigen::Vector3f& center, float radius, int depth, std::vector<int>& results) const {
		if (node == nullptr) {
			return;
		}

		const Eigen::Vector3f& point = points[node->index];
		float distance = (point - center).norm();
		if (distance <= radius) {
			results.push_back(node->index);
		}

		int axis = depth % 3;
		float diff = center[axis] - point[axis];

		if (diff <= radius) {
			radiusSearchRec(node->left, center, radius, depth + 1, results);
		}
		if (diff >= -radius) {
			radiusSearchRec(node->right, center, radius, depth + 1, results);
		}
	}

	void nearestNeighborRec(kdiNode* node, const Eigen::Vector3f& target, int depth, float& bestDist, int& bestIndex) const {
		if (node == nullptr) {
			return;
		}

		const Eigen::Vector3f& point = points[node->index];
		float distance = (point - target).squaredNorm();
		if (distance < bestDist) {
			bestDist = distance;
			bestIndex = node->index;
		}

		int axis = depth % 3;
		float diff = target[axis] - point[axis];

		kdiNode* nearChild = (diff < 0) ? node->left : node->right;
		kdiNode* farChild = (diff < 0) ? node->right : node->left;

		nearestNeighborRec(nearChild, target, depth + 1, bestDist, bestIndex);

		if (diff * diff < bestDist) {
			nearestNeighborRec(farChild, target, depth + 1, bestDist, bestIndex);
		}
	}

	void kNearestNeighborsRec(kdiNode* node, const Eigen::Vector3f& target, int depth, int k, std::vector<std::pair<float, int>>& neighbors) const {
		if (node == nullptr) {
			return;
		}

		const Eigen::Vector3f& point = points[node->index];
		float distance = (point - target).squaredNorm();

		if (neighbors.size() < k) {
			neighbors.emplace_back(distance, node->index);
		}
		else {
			auto maxIt = std::max_element(neighbors.begin(), neighbors.end());
			if (distance < maxIt->first) {
				*maxIt = std::make_pair(distance, node->index);
			}
		}

		int axis = depth % 3;
		float diff = target[axis] - point[axis];

		kdiNode* nearChild = (diff < 0) ? node->left : node->right;
		kdiNode* farChild = (diff < 0) ? node->right : node->left;

		kNearestNeighborsRec(nearChild, target, depth + 1, k, neighbors);

		auto maxIt = std::max_element(neighbors.begin(), neighbors.end());
		if (neighbors.size() < k || diff * diff < maxIt->first) {
			kNearestNeighborsRec(farChild, target, depth + 1, k, neighbors);
		}
	}
};


int main()
{
	App app;
	app.AddAppStartCallback([&](App* pApp) {
		auto renderer = pApp->GetRenderer();

		//TestManaged(renderer);

		{
			Processing::PatchProcessor pp;
			pp.Initialize(2000, 2000, 256 * 480);

			//for (int i = 3; i < 244; i++)
			int i = 3;
			//for (int i = 14; i < 15; i++)
			{
				auto te = Time::Now();
				LoadPatch(i, renderer);
				Time::End(te, "Loading PointCloud Patch");

				for (size_t i = 0; i < patchPoints.size(); i++)
				{
					auto& p = patchPoints[i];
					VisualDebugging::AddSphere("Original", p, { 0.1f, 0.1f, 0.1f }, { 0.0f, 0.0f, 0.0f }, Color4::White);
				}

				{
					float sampleSize = 0.1f;

					printf("[[[ patchPoints.size() ]]] : %llu\n", patchPoints.size());

					auto t = Time::Now();

					pp.DownSample(patchPoints.data(), patchPoints.size(), sampleSize, taabb.min(), taabb.max());

					t = Time::End(t, "DownSampling");

					printf("[[[ *pp.h_numberOfResultPoints ]]] : %llu\n", pp.h_numberOfResultPoints);

					for (size_t i = 0; i < pp.h_numberOfResultPoints; i++)
					{
						auto& p = pp.h_resultPoints[i];

						//p += Eigen::Vector3f(0.0f, 0.0f, 10.0f);

						VisualDebugging::AddSphere("DownSample", p, { 0.1f, 0.1f, 0.1f }, { 0.0f, 0.0f, 0.0f }, Color4::Blue);
					}
				}

				{
					float sampleSize = 0.1f;

					printf("[[[ patchPoints.size() ]]] : %llu\n", patchPoints.size());

					auto t = Time::Now();

					pp.MedianFilter(patchPoints.data(), patchPoints.size(), sampleSize, taabb.min(), taabb.max());

					t = Time::End(t, "MedianFilter");

					printf("[[[ *pp.h_numberOfResultPoints ]]] : %llu\n", pp.h_numberOfResultPoints);

					for (size_t i = 0; i < pp.h_numberOfResultPoints; i++)
					{
						auto& p = pp.h_resultPoints[i];

						//p += Eigen::Vector3f(0.0f, 0.0f, 10.0f);

						VisualDebugging::AddSphere("Filter", p, { 0.1f, 0.1f, 0.1f }, { 0.0f, 0.0f, 0.0f }, Color4::Red);
					}
				}
			}

			cout << taabb.min() << endl;
			cout << taabb.max() << endl;

			cout << taabb.max() - taabb.min() << endl;

			cout << "lmax : " << lmax.max() << endl;

			/*{
				printf("[[[ inputPoints.size() ]]] : %llu\n", inputPoints.size());

				Processing::PatchProcessor pp;
				pp.Initialize(2000, 2000, 256 * 480);

				pp.DeallocatedBuffers();

				auto t = Time::Now();

				pp.MedianFilter(inputPoints.data(), inputPoints.size(), sampleSize, taabb.min(), taabb.max());

				t = Time::End(t, "DownSample");

				printf("[[[ *pp.h_numberOfResultPoints ]]] : %llu\n", pp.h_numberOfResultPoints);

				for (size_t i = 0; i < pp.h_numberOfResultPoints; i++)
				{
					auto& p = pp.h_resultPoints[i];

					p += Eigen::Vector3f(0.0f, 10.0f, 0.0f);

					VisualDebugging::AddSphere("p", p, { 0.1f, 0.1f, 0.1f }, { 0.0f, 0.0f, 0.0f }, Color4::Red);
				}
			}*/


			//auto t = Time::Now();

			//Processing::PointCloud pointCloud;
			//cudaMalloc(&pointCloud.points, sizeof(Eigen::Vector3f) * inputPoints.size());
			//cudaMemcpy(pointCloud.points, inputPoints.data(), sizeof(Eigen::Vector3f) * inputPoints.size(), cudaMemcpyHostToDevice);
			//cudaDeviceSynchronize();

			//pointCloud.numberOfPoints = inputPoints.size();
			//pointCloud.pointMin = taabb.min();
			//pointCloud.pointMax = taabb.max();

			//auto result = Processing::DownSample(pointCloud, sampleSize);

			//t = Time::End(t, "DownSample");

			//for (size_t i = 0; i < result.numberOfPoints; i++)
			//{
			//	auto& p = result.points[i];

			//	VisualDebugging::AddSphere("p", p, { 0.05f, 0.05f, 0.05f }, { 0.0f, 0.0f, 0.0f }, Color4::Red);
			//}
		}


		return;

		//{
		//	auto t = Time::Now();
		//	AOctree::Test(inputPoints);
		//	Time::End(t, "AOCtree::Test");
		//}

		//{
		//	auto t = Time::Now();
		//	Algorithm::kdTree kdtree;
		//	Algorithm::kdNode* nodes = new Algorithm::kdNode[inputPoints.size()];
		//	for (size_t i = 0; i < inputPoints.size(); i++)
		//	{
		//		auto& p = inputPoints[i];
		//		nodes[i].x[0] = p.x();
		//		nodes[i].x[1] = p.y();
		//		nodes[i].x[2] = p.z();
		//	}
		//	kdtree.buildTree(nodes, inputPoints.size(), 0, 3);
		//	Time::End(t, "KDTree Build");
		//}

		//{
		//	auto t = Time::Now();
		//	Algorithm::kdiTree kdtree;
		//	kdtree.Init(inputPoints.data(), inputPoints.size());
		//	t = Time::End(t, "KDITree Initialize");
		//	kdtree.buildTree(kdtree.kdRoot, inputPoints.size(), 0, 3);
		//	t = Time::End(t, "KDITree Build");
		//	kdtree.findKNN({ 0.0f, 0.0f, 0.0f });
		//	t = Time::End(t, "findKNN");
		//	printf("visited : %d\n", kdtree.visited);
		//	for (size_t i = 0; i < kdtree.visited; i++)
		//	{
		//		auto node = kdtree.VisitedNodes[i];
		//		auto& p = inputPoints[node.index];
		//		VisualDebugging::AddSphere("Temp", p, { 1.0f, 1.0f, 1.0f }, { 0, 0, 0 }, Color4::Red);
		//	}
		//}

//#pragma region Gaussian Blur
//		//auto tg = Time::Now();
//		//auto blured = GaussianBlur(inputPoints, 1.0f, 1.0f);
//		//Time::End(tg, "GaussianBlur");
//
//		//vtkNew<vtkPoints> points;
//		//for (size_t i = 0; i < blured.size(); i++)
//		//{
//		//	auto& p = blured[i];
//		//	points->InsertNextPoint(p.x(), p.y(), p.z());
//		//}
//
//		//vtkNew<vtkPolyData> polyData;
//		//polyData->SetPoints(points);
//
//		//vtkNew<vtkVertexGlyphFilter> vertexFilter;
//		//vertexFilter->SetInputData(polyData);
//		//vertexFilter->Update();
//
//		//vtkNew<vtkPolyDataMapper> mapper;
//		//mapper->SetInputData(vertexFilter->GetOutput());
//
//		//vtkNew<vtkActor> actor;
//		//actor->SetMapper(mapper);
//
//		//actor->GetProperty()->SetPointSize(5.0f);
//		//actor->GetProperty()->SetColor(1.0, 0.0, 0.0);
//
//		//renderer->AddActor(actor);  
//#pragma endregion

		//{
		//	auto t = Time::Now();
		//	AOctree::Test(inputPoints);
		//	Time::End(t, "Test");
		//}

#pragma region Octree
		//auto t0 = Time::Now();

		//SVO::Octree* octree = new SVO::Octree;
		//pOctree = octree;
		//SVO::InitializeSVO(octree, inputPoints.size() * 13);

		//Time::End(t0, "Initialize Octree");

		//auto te = Time::Now();
		//SVO::IntegratePointCloud(octree, inputPoints.data(), inputPoints.size(), 500.0f, 13);
		//Time::End(te, "Integrate PointCloud Patches");
		//
		//printf("inputPoints.size() : %llu, octreeNodeCount : %llu\n", inputPoints.size(), octree->nodeBufferIndex);

		////SVO::TraverseOctree(octree, [](const SVO::SVONode& node, int currentDepth) {
		////	float halfSize = node.size * 0.5f;

		////	stringstream ss;
		////	ss << "WiredBox_" << currentDepth;
		////	VisualDebugging::AddWiredBox(
		////		ss.str(),
		////		node.center - Eigen::Vector3f(halfSize, halfSize, halfSize),
		////		node.center + Eigen::Vector3f(halfSize, halfSize, halfSize),
		////		Color4::White);
		////	});

		////{
		////	SVO::NearestNeighborResult result;
		////	SVO::NearestNeighborDFS(octree, octree->rootIndex, { 3.0f, 3.0f, 3.0f }, result);
		////}

		////vector<SVO::Triangle> triangles;
		////SVO::ExtractTrianglesFromOctree(octree, triangles);

		////// Step 5: Output the resulting triangles
		////for (const auto& triangle : triangles)
		////{
		////	std::cout << "Triangle vertices:" << std::endl;
		////	for (const auto& vertex : triangle.vertices)
		////	{
		////		std::cout << "(" << vertex.x() << ", " << vertex.y() << ", " << vertex.z() << ")" << std::endl;
		////	}
		////	std::cout << "-------------------" << std::endl;
		////}
#pragma endregion

#if 0
		{
			auto t = Time::Now();
			kdiTree tree(inputPoints.data(), inputPoints.size());
			t = Time::End(t, "Building KDTree");

			vector<int> results;

#pragma region Radius Search
			//results.clear();
			//t = Time::Now();
			//tree.radiusSearch({ 0.0f, 0.0f, 0.0f }, 7.0f, results);
			//for (size_t i = 0; i < results.size(); i++)
			//{
			//	auto& p = inputPoints[results[i]];
			//	VisualDebugging::AddSphere("Temp", p, { 0.1f, 0.1f, 0.1f }, { 0, 0, 0 }, Color4::Red);
			//}
			//t = Time::End(t, "Radius Search");
#pragma endregion



#pragma region Range Search
			//results.clear();
			//t = Time::Now();
			//tree.rangeSearch({ 0.0f, 0.0f, 0.0f }, { 10.0f, 10.0f, 10.0f }, results);
			//t = Time::End(t, "Range Search");

			//for (size_t i = 0; i < results.size(); i++)
			//{
			//	auto& p = inputPoints[results[i]];
			//	VisualDebugging::AddSphere("Temp", p, { 0.1f, 0.1f, 0.1f }, { 0, 0, 0 }, Color4::Red);
			//}
#pragma endregion

			int k = 300;
			Eigen::Vector3f target(3.0f, 3.0f, 3.0f);

#pragma region Nearest Neighbor Search
			//t = Time::Now();
			//// 최근접 이웃 검색 테스트
			//int nearestIndex = tree.nearestNeighbor(target);
			//std::cout << "\nNearest Neighbor Result:" << std::endl;
			//t = Time::End(t, "Nearest Neighbor Search");
			//VisualDebugging::AddSphere("Temp", inputPoints[nearestIndex], { 1.0f, 1.0f, 1.0f }, { 0, 0, 0 }, Color4::Blue);
#pragma endregion

#pragma region K - Nearest Neighbor Search
			t = Time::Now();
			// k-최근접 이웃 검색 테스트
			results.clear();
			t = Time::Now();
			tree.kNearestNeighbors(target, k, results);
			t = Time::End(t, "K Nearest Neighbor Search");

			printf("results.size() : %d\n", results.size());

			for (size_t i = 0; i < results.size(); i++)
			{
				auto& p = inputPoints[results[i]];
				VisualDebugging::AddSphere("Temp", p, { 0.1f, 0.1f, 0.1f }, { 0, 0, 0 }, Color4::Yellow);
			}
#pragma endregion

#pragma region K - Nearest Neighbor Search
			t = Time::Now();
			// k-최근접 이웃 검색 테스트
			results.clear();
			t = Time::Now();
			tree.kNearestNeighborsWithinRadius(target, k, 2.0f, results);
			t = Time::End(t, "K Nearest Neighbor within Radius Search");

			printf("results.size() : %d\n", results.size());

			for (size_t i = 0; i < results.size(); i++)
			{
				auto& p = inputPoints[results[i]];
				VisualDebugging::AddSphere("Temp", p, { 0.1f, 0.1f, 0.1f }, { 0, 0, 0 }, Color4::Blue);
			}
#pragma endregion

#pragma region K - Nearest Neighbor Search
			t = Time::Now();
			// k-최근접 이웃 검색 테스트
			results.clear();
			t = Time::Now();
			tree.kNearestNeighborsWithinRange(target, k, { 2.0, 2.0, 2.0f }, { 5.0, 5.0f, 5.0f }, results);
			t = Time::End(t, "K Nearest Neighbor within Range Search");

			printf("results.size() : %d\n", results.size());

			for (size_t i = 0; i < results.size(); i++)
			{
				auto& p = inputPoints[results[i]];
				VisualDebugging::AddSphere("Temp", p, { 0.1f, 0.1f, 0.1f }, { 0, 0, 0 }, Color4::Cyan);
			}
#pragma endregion
		}
#endif

		VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 100.0f, 0.0f, 0.0f }, Color4::Red);
		VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 100.0f, 0.0f }, Color4::Green);
		VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, 100.0f }, Color4::Blue);
		});

	app.AddKeyPressCallback(OnKeyPress);
	app.Run();

	return 0;
}
