#include <Common.h>
#include <App/App.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include <Algorithm/Octree.hpp>

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
		VisualDebugging::ToggleVisibility("OctreeNode");
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

		for (int i = 0; i < 14; i++)
		{
			stringstream ss;
			ss << "Cubes_" << i;
			VisualDebugging::SetVisibility(ss.str(), false);
		}
		{
			stringstream ss;
			ss << "Cubes_" << depthIndex;
			VisualDebugging::SetVisibility(ss.str(), true);
		}
	}
	else if (key == "Right")
	{
		depthIndex++;
		if (depthIndex > 14) depthIndex = 13;

		for (int i = 0; i < 14; i++)
		{
			stringstream ss;
			ss << "Cubes_" << i;
			VisualDebugging::SetVisibility(ss.str(), false);
		}
		{
			stringstream ss;
			ss << "Cubes_" << depthIndex;
			VisualDebugging::SetVisibility(ss.str(), true);
		}
	}
	else if (key == "space")
	{
	}
}


namespace Test
{
	// 삼각형을 생성할 에지 배열 (Marching Cubes의 사례별 표 사용)
	static const int edgeTable[256] = {
		0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
		0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
		0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
		0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
		0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
		0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
		0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
		0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
		0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
		0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
		0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
		0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
		0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
		0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
		0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
		0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
		0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
		0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
		0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
		0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
		0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
		0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
		0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
		0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
		0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
		0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
		0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
		0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
		0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
		0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
		0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
		0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0 };

	static const int triTable[256][16] = {
		{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
		{3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
		{3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
		{3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
		{9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
		{9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
		{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
		{8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
		{9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
		{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
		{3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
		{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
		{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
		{4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
		{5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
		{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
		{9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
		{0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
		{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
		{10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
		{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
		{5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
		{9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
		{0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
		{1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
		{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
		{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
		{2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
		{7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
		{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
		{11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
		{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
		{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
		{11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
		{1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
		{9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
		{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
		{2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
		{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
		{6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
		{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
		{6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
		{5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
		{1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
		{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
		{6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
		{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
		{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
		{3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
		{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
		{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
		{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
		{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
		{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
		{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
		{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
		{10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
		{10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
		{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
		{1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
		{0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
		{10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
		{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
		{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
		{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
		{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
		{3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
		{6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
		{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
		{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
		{10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
		{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
		{7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
		{7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
		{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
		{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
		{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
		{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
		{0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
		{7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
		{10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
		{2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
		{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
		{7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
		{2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
		{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
		{10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
		{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
		{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
		{7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
		{6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
		{8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
		{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
		{6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
		{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
		{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
		{8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
		{0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
		{1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
		{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
		{10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
		{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
		{10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
		{5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
		{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
		{9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
		{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
		{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
		{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
		{7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
		{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
		{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
		{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
		{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
		{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
		{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
		{6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
		{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
		{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
		{6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
		{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
		{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
		{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
		{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
		{9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
		{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
		{1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
		{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
		{0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
		{5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
		{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
		{11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
		{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
		{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
		{2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
		{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
		{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
		{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
		{1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
		{9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
		{9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
		{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
		{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
		{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
		{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
		{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
		{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
		{9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
		{5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
		{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
		{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
		{8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
		{0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
		{9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
		{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
		{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
		{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
		{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
		{11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
		{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
		{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
		{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
		{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
		{1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
		{4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
		{3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
		{0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
		{9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
		{1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1} };

	// TSDF value container for each node
	struct TSDFValue {
		float tsdf;    // TSDF value
		float weight;  // Accumulated weight

		TSDFValue() : tsdf(FLT_MAX), weight(0.0f) {}
	};

	// Octree Node Class
	class OctreeNode {
	public:
		TSDFValue tsdfValue;
		std::array<std::unique_ptr<OctreeNode>, 8> children;

		OctreeNode() {
			for (int i = 0; i < 8; ++i) {
				children[i] = nullptr;
			}
		}

		bool isLeaf() const {
			for (const auto& child : children) {
				if (child != nullptr) {
					return false;
				}
			}
			return true;
		}
	};

	// Octree Class Definition
	class Octree {
	public:
		Octree(float size, Eigen::Vector3f origin, int maxDepth)
			: root(std::make_unique<OctreeNode>()), size(size), origin(origin), maxDepth(maxDepth) {}

		void integrate(const std::vector<Eigen::Vector3f>& inputPoints, const Eigen::Vector3f& cameraPos, float truncationDistance) {
			this->truncationDistance = truncationDistance; // Store truncation distance as a member variable
			for (const auto& point : inputPoints) {
				integratePoint(root.get(), point, cameraPos, origin, size, 0);
			}
		}

		void traverse(std::function<void(const OctreeNode&, const Eigen::Vector3f&, float, int)> visitFunc) const {
			traverseNode(root.get(), origin, size, 0, visitFunc);
		}

		void extractTriangles(std::vector<Eigen::Vector3f>& vertices, std::vector<Eigen::Vector3i>& triangles) {
			extractTrianglesFromNode(root.get(), origin, size, 0, vertices, triangles);
		}

	private:
		std::unique_ptr<OctreeNode> root;
		float size;
		Eigen::Vector3f origin;
		int maxDepth;
		float truncationDistance; // Member variable to store truncation distance

		void integratePoint(OctreeNode* node, const Eigen::Vector3f& point, const Eigen::Vector3f& cameraPos,
			const Eigen::Vector3f& nodeOrigin, float nodeSize, int depth) {
			if (depth == maxDepth) {
				// Compute the signed distance from the voxel center to the surface point
				Eigen::Vector3f voxelCenter = nodeOrigin + Eigen::Vector3f(nodeSize / 2.0f, nodeSize / 2.0f, nodeSize / 2.0f);
				float distance = (point - voxelCenter).norm();
				float signedDistance = (voxelCenter - point).dot((point - cameraPos).normalized()); // Signed distance

				float tsdf = std::max(-1.0f, std::min(1.0f, signedDistance / truncationDistance));
				float weight = 1.0f;

				// Fuse TSDF value and weight into the voxel
				node->tsdfValue.tsdf = (node->tsdfValue.tsdf * node->tsdfValue.weight + tsdf * weight) / (node->tsdfValue.weight + weight);
				node->tsdfValue.weight += weight;
				return;
			}

			// Ensure children exist (subdivide) if the node is not at max depth
			if (node->isLeaf()) {
				subdivide(node);
			}

			// Determine which child node the point belongs to and continue integration recursively
			int childIndex = getChildIndex(nodeOrigin, nodeSize, point);
			Eigen::Vector3f childOrigin = computeChildOrigin(nodeOrigin, nodeSize, childIndex);
			float childSize = nodeSize / 2.0f;

			integratePoint(node->children[childIndex].get(), point, cameraPos, childOrigin, childSize, depth + 1);
		}

		void traverseNode(const OctreeNode* node, const Eigen::Vector3f& nodeOrigin, float nodeSize, int depth,
			std::function<void(const OctreeNode&, const Eigen::Vector3f&, float, int)> visitFunc) const {
			if (node == nullptr) return;

			visitFunc(*node, nodeOrigin, nodeSize, depth);

			if (!node->isLeaf()) {
				float childSize = nodeSize / 2.0f;
				for (int i = 0; i < 8; ++i) {
					Eigen::Vector3f childOrigin = computeChildOrigin(nodeOrigin, nodeSize, i);
					traverseNode(node->children[i].get(), childOrigin, childSize, depth + 1, visitFunc);
				}
			}
		}

		void extractTrianglesFromNode(OctreeNode* node, const Eigen::Vector3f& nodeOrigin, float nodeSize, int depth,
			std::vector<Eigen::Vector3f>& vertices, std::vector<Eigen::Vector3i>& triangles) {
			if (node == nullptr) return;

			if (!node->isLeaf()) {
				float childSize = nodeSize / 2.0f;
				for (int i = 0; i < 8; ++i) {
					Eigen::Vector3f childOrigin = computeChildOrigin(nodeOrigin, nodeSize, i);
					extractTrianglesFromNode(node->children[i].get(), childOrigin, childSize, depth + 1, vertices, triangles);
				}
			}
			else
			{
				// Define corner points
				std::array<Eigen::Vector3f, 8> cornerPoints = {
					nodeOrigin,
					nodeOrigin + Eigen::Vector3f(nodeSize, 0, 0),
					nodeOrigin + Eigen::Vector3f(nodeSize, nodeSize, 0),
					nodeOrigin + Eigen::Vector3f(0, nodeSize, 0),
					nodeOrigin + Eigen::Vector3f(0, 0, nodeSize),
					nodeOrigin + Eigen::Vector3f(nodeSize, 0, nodeSize),
					nodeOrigin + Eigen::Vector3f(nodeSize, nodeSize, nodeSize),
					nodeOrigin + Eigen::Vector3f(0, nodeSize, nodeSize)
				};

				// Get TSDF values at corners
				std::array<float, 8> tsdfValues;
				for (int i = 0; i < 8; ++i) {
					Eigen::Vector3f voxelCenter = cornerPoints[i] + Eigen::Vector3f(nodeSize / 2.0f, nodeSize / 2.0f, nodeSize / 2.0f);
					float distance = (voxelCenter - origin).norm();
					float sdf = (distance - (cornerPoints[i] - origin).norm()); // Signed distance to the surface
					tsdfValues[i] = std::max(-1.0f, std::min(1.0f, sdf / truncationDistance));
				}

				// Calculate cube index
				int cubeIndex = 0;
				for (int i = 0; i < 8; ++i) {
					if (tsdfValues[i] < 0) {
						cubeIndex |= (1 << i);
					}
				}

				if (edgeTable[cubeIndex] == 0) {
					return;
				}

				// Interpolate vertices along edges
				std::array<Eigen::Vector3f, 12> edgeVertices;
				if (edgeTable[cubeIndex] & 1) {
					edgeVertices[0] = interpolate(cornerPoints[0], cornerPoints[1], tsdfValues[0], tsdfValues[1]);
				}
				if (edgeTable[cubeIndex] & 2) {
					edgeVertices[1] = interpolate(cornerPoints[1], cornerPoints[2], tsdfValues[1], tsdfValues[2]);
				}
				if (edgeTable[cubeIndex] & 4) {
					edgeVertices[2] = interpolate(cornerPoints[2], cornerPoints[3], tsdfValues[2], tsdfValues[3]);
				}
				if (edgeTable[cubeIndex] & 8) {
					edgeVertices[3] = interpolate(cornerPoints[3], cornerPoints[0], tsdfValues[3], tsdfValues[0]);
				}
				if (edgeTable[cubeIndex] & 16) {
					edgeVertices[4] = interpolate(cornerPoints[4], cornerPoints[5], tsdfValues[4], tsdfValues[5]);
				}
				if (edgeTable[cubeIndex] & 32) {
					edgeVertices[5] = interpolate(cornerPoints[5], cornerPoints[6], tsdfValues[5], tsdfValues[6]);
				}
				if (edgeTable[cubeIndex] & 64) {
					edgeVertices[6] = interpolate(cornerPoints[6], cornerPoints[7], tsdfValues[6], tsdfValues[7]);
				}
				if (edgeTable[cubeIndex] & 128) {
					edgeVertices[7] = interpolate(cornerPoints[7], cornerPoints[4], tsdfValues[7], tsdfValues[4]);
				}
				if (edgeTable[cubeIndex] & 256) {
					edgeVertices[8] = interpolate(cornerPoints[0], cornerPoints[4], tsdfValues[0], tsdfValues[4]);
				}
				if (edgeTable[cubeIndex] & 512) {
					edgeVertices[9] = interpolate(cornerPoints[1], cornerPoints[5], tsdfValues[1], tsdfValues[5]);
				}
				if (edgeTable[cubeIndex] & 1024) {
					edgeVertices[10] = interpolate(cornerPoints[2], cornerPoints[6], tsdfValues[2], tsdfValues[6]);
				}
				if (edgeTable[cubeIndex] & 2048) {
					edgeVertices[11] = interpolate(cornerPoints[3], cornerPoints[7], tsdfValues[3], tsdfValues[7]);
				}

				// Create triangles from the vertices using the triTable
				for (int i = 0; triTable[cubeIndex][i] != -1; i += 3) {
					Eigen::Vector3f v1 = edgeVertices[triTable[cubeIndex][i]];
					Eigen::Vector3f v2 = edgeVertices[triTable[cubeIndex][i + 1]];
					Eigen::Vector3f v3 = edgeVertices[triTable[cubeIndex][i + 2]];

					vertices.push_back(v1);
					vertices.push_back(v2);
					vertices.push_back(v3);

					int vertexOffset = (int)vertices.size() - 3;
					triangles.emplace_back(vertexOffset, vertexOffset + 1, vertexOffset + 2);
				}
			}
		}

		Eigen::Vector3f interpolate(const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, float tsdf1, float tsdf2) {
			if (std::abs(tsdf1 - tsdf2) > 1e-5) {
				float t = -tsdf1 / (tsdf2 - tsdf1);
				return p1 + t * (p2 - p1);
			}
			return p1;
		}

		Eigen::Vector3f computeChildOrigin(const Eigen::Vector3f& nodeOrigin, float nodeSize, int childIndex) const {
			float halfSize = nodeSize / 2.0f;
			Eigen::Vector3f childOrigin = nodeOrigin;

			if (childIndex & 1) childOrigin.x() += halfSize;
			if (childIndex & 2) childOrigin.y() += halfSize;
			if (childIndex & 4) childOrigin.z() += halfSize;

			return childOrigin;
		}

		void subdivide(OctreeNode* node) {
			for (int i = 0; i < 8; ++i) {
				node->children[i] = std::make_unique<OctreeNode>();
			}
		}

		int getChildIndex(const Eigen::Vector3f& nodeOrigin, float nodeSize, const Eigen::Vector3f& point) {
			int index = 0;
			if (point.x() > nodeOrigin.x() + nodeSize / 2) index |= 1;
			if (point.y() > nodeOrigin.y() + nodeSize / 2) index |= 2;
			if (point.z() > nodeOrigin.z() + nodeSize / 2) index |= 4;
			return index;
		}
	};
}

int main()
{
	App app;
	app.AddAppStartCallback([&](App* pApp) {
		auto renderer = pApp->GetRenderer();

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
					VisualDebugging::AddSphere("Original", p, { 0.05f, 0.05f, 0.05f }, { 0.0f, 0.0f, 0.0f }, Color4::White);
				}

				//{
				//	float sampleSize = 0.1f;

				//	printf("[[[ patchPoints.size() ]]] : %llu\n", patchPoints.size());

				//	auto t = Time::Now();

				//	pp.DownSample(patchPoints.data(), patchPoints.size(), sampleSize, taabb.min(), taabb.max());

				//	t = Time::End(t, "DownSampling");

				//	printf("[[[ *pp.h_numberOfResultPoints ]]] : %llu\n", pp.h_numberOfResultPoints);

				//	for (size_t i = 0; i < pp.h_numberOfResultPoints; i++)
				//	{
				//		auto& p = pp.h_resultPoints[i];

				//		//p += Eigen::Vector3f(0.0f, 0.0f, 10.0f);

				//		VisualDebugging::AddSphere("DownSample", p, { 0.05f, 0.05f, 0.05f }, { 0.0f, 0.0f, 0.0f }, Color4::Blue);
				//	}
				//}

				//{
				//	float sampleSize = 0.1f;

				//	printf("[[[ patchPoints.size() ]]] : %llu\n", patchPoints.size());

				//	auto t = Time::Now();

				//	pp.MedianFilter(patchPoints.data(), patchPoints.size(), sampleSize, taabb.min(), taabb.max());

				//	t = Time::End(t, "MedianFilter");

				//	printf("[[[ *pp.h_numberOfResultPoints ]]] : %llu\n", pp.h_numberOfResultPoints);

				//	for (size_t i = 0; i < pp.h_numberOfResultPoints; i++)
				//	{
				//		auto& p = pp.h_resultPoints[i];

				//		//p += Eigen::Vector3f(0.0f, 0.0f, 10.0f);

				//		VisualDebugging::AddSphere("Filter", p, { 0.05f, 0.05f, 0.05f }, { 0.0f, 0.0f, 0.0f }, Color4::Red);
				//	}
				//}

				{
					float sampleSize = 0.1f;

					printf("[[[ patchPoints.size() ]]] : %llu\n", patchPoints.size());

					auto t = Time::Now();

					pp.MarchingSquare(patchPoints.data(), patchPoints.size(), sampleSize, taabb.min(), taabb.max());

					t = Time::End(t, "MarchingSquare");

					printf("[[[ *pp.h_numberOfTriangleVertices ]]] : %llu\n", pp.h_numberOfTriangleVertices);
					printf("[[[ *pp.h_numberOfTriangleIndices ]]] : %llu\n", pp.h_numberOfTriangleIndices);

					//for (size_t i = 0; i < pp.h_numberOfTriangleIndices / 3; i++)
					//{
					//	auto ti0 = pp.h_triangleIndices[i * 3];
					//	auto ti1 = pp.h_triangleIndices[i * 3 + 1];
					//	auto ti2 = pp.h_triangleIndices[i * 3 + 2];

					//	auto& p0 = pp.h_triangleVertices[ti0];
					//	auto& p1 = pp.h_triangleVertices[ti1];
					//	auto& p2 = pp.h_triangleVertices[ti2];

					//	VisualDebugging::AddTriangle("Triangles", p0, p1, p2, Color4::White);
					//}

					for (size_t i = 0; i < pp.h_numberOfResultPoints; i++)
					{
						auto& p = pp.h_resultPoints[i];

						//p += Eigen::Vector3f(0.0f, 0.0f, 10.0f);

						VisualDebugging::AddSphere("Filter", p, { 0.05f, 0.05f, 0.05f }, { 0.0f, 0.0f, 0.0f }, Color4::Red);
					}
				}

				cout << taabb.min() << endl;
				cout << taabb.max() << endl;

				cout << taabb.max() - taabb.min() << endl;

				cout << "lmax : " << lmax.max() << endl;

				return;

				//Spatial::Octree octree;
				//octree.Initialize(inputPoints.size() * 13);

				//auto t = Time::Now();
				//for (size_t i = 0; i < inputPoints.size(); i++)
				//{
				//	octree.Insert(inputPoints[i]);
				//}
				//t = Time::End(t, "Octree");

				//octree.Traverse(octree.root, [](Spatial::OctreeNode& node) {
				//	//cout << node.min.transpose() << endl;
				//	//cout << node.max.transpose() << endl;

				//	VisualDebugging::AddCube("Cubes", (node.max + node.min) * 0.5f, (node.max - node.min), {0.0f, 0.0f, 0.0f}, Color4::White);
				//	});

				//Spatial::Octree octree(Eigen::Vector3f::Zero(), 50);
				Spatial::Octree octree(taabb.center(), 50, inputPoints.size() * 11);
				auto t = Time::Now();
				for (size_t i = 0; i < inputPoints.size(); i++)
				{
					auto& p = inputPoints[i];
					// Assuming you have some surface point or sensor origin, calculate the distance.
					// Example: using an arbitrary distance function to compute the TSDF
					float surface_distance = (p - taabb.center()).norm();  // Replace with correct distance to surface or origin
					float truncation_distance = 5.0f;  // Example truncation distance, adjust as needed

					// Calculate the TSDF value
					float tsdf = std::max(-1.0f, std::min(1.0f, surface_distance / truncation_distance));

					tsdf = 0.0f;

					// Integrate the TSDF value into the octree
					octree.integrate_point(p, tsdf, 1.0f);  // Use computed TSDF value, not 0.0f
				}
				t = Time::End(t, "Octree");

				size_t count = 0;
				int maxDepth = 0;
				octree.traverse([&](const Spatial::OctreeNode* node, int depth) {
					count++;
					if (maxDepth < depth) maxDepth = depth;
					if (node->is_leaf && fabsf(node->tsdf_value) < 0.025f)
					{
						stringstream ss;
						ss << "Cubes_" << depth;
						//VisualDebugging::AddCube(ss.str(), node->center, { node->nodeSize * 2.0f, node->nodeSize * 2.0f, node->nodeSize * 2.0f }, { 0.0f, 0.0f, 0.0f }, Color4::White);
					}
					});

				printf("count : %llu\n", count);
				printf("maxDepth : %d\n", maxDepth);
				printf("inputPoints.size() : %llu\n", inputPoints.size());

				return;

				VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 100.0f, 0.0f, 0.0f }, Color4::Red);
				VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 100.0f, 0.0f }, Color4::Green);
				VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, 100.0f }, Color4::Blue);
			}
		}
		});

	app.AddKeyPressCallback(OnKeyPress);
	app.Run();

	return 0;
}
