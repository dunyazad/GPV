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
					printf("octree->maxDepth : %llu\n", octree->maxDepth);
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
Eigen::AlignedBox3f aabb_0(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));
Eigen::AlignedBox3f aabb_45(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));
Eigen::AlignedBox3f taabb(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));
Eigen::AlignedBox3f lmax(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));

vector<Eigen::Vector3f> patchPoints_0;
vector<Eigen::Vector3f> patchPoints_45;
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

	aabb_0 = Eigen::AlignedBox3f(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));
	aabb_45 = Eigen::AlignedBox3f(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));

	int count = 0;

	patchPoints_0.clear();
	patchPoints_45.clear();

	for (size_t i = 0; i < size_0; i++)
	{
		auto& p = points_0[i];

		if ((p.x() > -1000 && p.y() > -1000 && p.z() > -1000) &&
			(p.x() < 1000 && p.y() < 1000 && p.z() < 1000))
		{
			patchPoints_0.push_back(p);
			
			Eigen::Vector4f p4(p.x(), p.y(), p.z(), 1.0f);
			Eigen::Vector4f tp = t0 * p4;
			Eigen::Vector3f tp3 = Eigen::Vector3f(tp.x(), tp.y(), tp.z());

			points->InsertNextPoint(tp3.data());

			aabb_0.extend(tp3);
			taabb.extend(tp3);

			lmax.extend(aabb_0.max() - aabb_0.min());

			inputPoints.push_back(tp3);

			count++;
		}
	}

	for (size_t i = 0; i < size_45; i++)
	{
		auto& p = points_45[i];

		if ((p.x() > -1000 && p.y() > -1000 && p.z() > -1000) &&
			(p.x() < 1000 && p.y() < 1000 && p.z() < 1000))
		{
			patchPoints_45.push_back(p);
	
			Eigen::Vector4f p4(p.x(), p.y(), p.z(), 1.0f);
			Eigen::Vector4f tp = t45 * p4;
			Eigen::Vector3f tp3 = Eigen::Vector3f(tp.x(), tp.y(), tp.z());

			points->InsertNextPoint(tp3.data());

			aabb_45.extend(tp3);
			taabb.extend(tp3);

			lmax.extend(aabb_45.max() - aabb_45.min());
	
			inputPoints.push_back(tp3);

			count++;
		}
	}

	std::cout << aabb_0.min() << std::endl;
	std::cout << aabb_0.max() << std::endl;
	std::cout << aabb_45.min() << std::endl;
	std::cout << aabb_45.max() << std::endl;

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

struct Voxel
{
	Eigen::Vector3f position = { FLT_MAX, FLT_MAX, FLT_MAX };
	Eigen::Vector3f normal = { FLT_MAX, FLT_MAX, FLT_MAX };
	float tsdf = FLT_MAX;
	float weight = 0.0f;
};

struct VoxelKey
{
	int x = 0;
	int y = 0;
	int z = 0;

	bool operator==(const VoxelKey& other) const {
		return x == other.x && y == other.y && z == other.z;
	}
};

// std::hash 특수화
//namespace std {
//	template <>
//	struct hash<VoxelKey> {
//		std::size_t operator()(const VoxelKey& key) const {
//			// 해시 값을 계산하는 방법 정의 (단순 예제)
//			return ((std::hash<int>()(key.x) ^ (std::hash<int>()(key.y) << 1)) >> 1) ^ (std::hash<int>()(key.z) << 1);
//		}
//	};
//}

namespace std {
	template <>
	struct hash<VoxelKey> {
		std::size_t operator()(const VoxelKey& key) const {
			// 해시 값을 생성할 때 seed 값을 활용하여 결합
			std::size_t seed = 0;
			hash_combine(seed, key.x);
			hash_combine(seed, key.y);
			hash_combine(seed, key.z);
			return seed;
		}

		// 해시 결합을 위한 함수 (boost 라이브러리의 해시 결합과 비슷하게 구현)
		static void hash_combine(std::size_t& seed, int value) {
			// 정수 해싱에 적합한 'magic number'와 XOR, bitwise 연산을 사용하여 해시값을 결합
			std::hash<int> hasher;
			seed ^= hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		}
	};
}

unordered_map<VoxelKey, Voxel> voxels;

VoxelKey GetKey(const Eigen::Vector3f& position, float voxelSize = 0.1f)
{
	int x = static_cast<int>(floorf(position.x() / voxelSize));
	int y = static_cast<int>(floorf(position.y() / voxelSize));
	int z = static_cast<int>(floorf(position.z() / voxelSize));
	return { x, y, z };
}

Eigen::Vector3f GetPosition(const VoxelKey& key, float voxelSize = 0.1f)
{
	return { (float)key.x * voxelSize, (float)key.y * voxelSize, (float)key.z * voxelSize };
}

int main()
{
	App app;
	app.AddAppStartCallback([&](App* pApp) {
		auto renderer = pApp->GetRenderer();

		auto t = Time::Now();

		CUDA::cuCache cache;
		t = Time::End(t, "Cache Initialize");
		
		vector<CUDA::Voxel> h_cache(cache.cacheSize.x * cache.cacheSize.y * cache.cacheSize.z);

		{
			Eigen::Vector3f* points;
			cudaMalloc(&points, sizeof(Eigen::Vector3f) * 256 * 480);

			//Processing::PatchProcessor pp;
			//pp.Initialize(2000, 2000, 256 * 480);

			//int i = 3;
			//for (int i = 3; i < 244; i++)
			for (int i = 3; i < 57; i++)
			//for (int cnt = 0; cnt < 4; cnt++)
			{
				auto te = Time::Now();
				LoadPatch(i, renderer);
				Time::End(te, "Loading PointCloud Patch");

				//for (size_t i = 0; i < patchPoints_0.size(); i++)
				//{
				//	auto& p = patchPoints_0[i];
				//	VisualDebugging::AddSphere("Original_0", p, { 0.05f, 0.05f, 0.05f }, { 0.0f, 0.0f, 0.0f }, Color4::Red);
				//}
				//for (size_t i = 0; i < patchPoints_45.size(); i++)
				//{
				//	auto& p = patchPoints_45[i];
				//	VisualDebugging::AddSphere("Original_45", p, { 0.05f, 0.05f, 0.05f }, { 0.0f, 0.0f, 0.0f }, Color4::Blue);
				//}

				{
					cudaMemcpy(points, patchPoints_0.data(), sizeof(Eigen::Vector3f) * patchPoints_0.size(), cudaMemcpyHostToDevice);
					cudaDeviceSynchronize();

					auto cp = Eigen::Vector3f(transform_0[12], transform_0[13], transform_0[14] + 20.0f);

					float minx = floorf(aabb_0.min().x() / cache.voxelSize) * cache.voxelSize;
					float miny = floorf(aabb_0.min().y() / cache.voxelSize) * cache.voxelSize;
					float minz = floorf(aabb_0.min().z() / cache.voxelSize) * cache.voxelSize;

					auto cacheMinKey = GetKey({ minx, miny, minz });

					cache.ClearCache();

					auto t = Time::Now();
					for (int z = 0; z < cache.cacheSize.z; z++)
					{
						for (int y = 0; y < cache.cacheSize.y; y++)
						{
							for (int x = 0; x < cache.cacheSize.x; x++)
							{
								VoxelKey key{ x + cacheMinKey.x, y + cacheMinKey.y, z + cacheMinKey.z };
								if (0 != voxels.count(key))
								{
									auto& voxel = voxels[key];

									int index = z * cache.cacheSize.z * cache.cacheSize.y + y * cache.cacheSize.x + x;
									cache.cache[index].tsdfValue = voxel.tsdf;
									cache.cache[index].weight = voxel.weight;
								}
							}
						}
					}
					t = Time::End(t, "Fetching");

					t = Time::Now();
					cache.Integrate({minx, miny, minz}, cp, Eigen::Matrix4f(transform_0), inputPoints.size(), points, nullptr, nullptr);
					Time::End(te, "Cache Integrate 0");

					for (int z = 0; z < cache.cacheSize.z; z++)
					{
						for (int y = 0; y < cache.cacheSize.y; y++)
						{
							for (int x = 0; x < cache.cacheSize.x; x++)
							{
								int index = z * cache.cacheSize.z * cache.cacheSize.y + y * cache.cacheSize.x + x;
								auto voxel = cache.cache[index];
								//auto voxel = h_cache[index];

								if (voxel.weight > 0.0f)
									//if (-0.05f < voxel.tsdfValue && voxel.tsdfValue < 0.05f)
								{
									//Eigen::Vector3f position = Eigen::Vector3f(
									//	x * cache.voxelSize + cache.cacheMin.x(),
									//	y * cache.voxelSize + cache.cacheMin.y(),
									//	z * cache.voxelSize + cache.cacheMin.z());

									//auto key = GetKey(position);
									auto key = VoxelKey{ x + cacheMinKey.x, y + cacheMinKey.y, z + cacheMinKey.z };
									voxels[key].tsdf = voxel.tsdfValue;
									voxels[key].weight = voxel.weight;
								}
							}
						}
					}
				}

				{
					cudaMemcpy(points, patchPoints_45.data(), sizeof(Eigen::Vector3f) * patchPoints_45.size(), cudaMemcpyHostToDevice);
					cudaDeviceSynchronize();

					auto cp = Eigen::Vector3f(transform_45[12], transform_45[13], transform_45[14] + 20.0f);

					float minx = floorf(aabb_45.min().x() / cache.voxelSize) * cache.voxelSize;
					float miny = floorf(aabb_45.min().y() / cache.voxelSize) * cache.voxelSize;
					float minz = floorf(aabb_45.min().z() / cache.voxelSize) * cache.voxelSize;

					auto cacheMinKey = GetKey({ minx, miny, minz });

					cache.ClearCache();

					auto t = Time::Now();
					for (int z = 0; z < cache.cacheSize.z; z++)
					{
						for (int y = 0; y < cache.cacheSize.y; y++)
						{
							for (int x = 0; x < cache.cacheSize.x; x++)
							{
								VoxelKey key{ x + cacheMinKey.x, y + cacheMinKey.y, z + cacheMinKey.z };
								if (0 != voxels.count(key))
								{
									auto& voxel = voxels[key];

									int index = z * cache.cacheSize.z * cache.cacheSize.y + y * cache.cacheSize.x + x;
									cache.cache[index].tsdfValue = voxel.tsdf;
									cache.cache[index].weight = voxel.weight;
								}
							}
						}
					}
					t = Time::End(t, "Fetching");

					t = Time::Now();
					cache.Integrate({ minx, miny, minz }, cp, Eigen::Matrix4f(transform_45), inputPoints.size(), points, nullptr, nullptr);
					Time::End(te, "Cache Integrate 45");

					for (int z = 0; z < cache.cacheSize.z; z++)
					{
						for (int y = 0; y < cache.cacheSize.y; y++)
						{
							for (int x = 0; x < cache.cacheSize.x; x++)
							{
								int index = z * cache.cacheSize.z * cache.cacheSize.y + y * cache.cacheSize.x + x;
								auto voxel = cache.cache[index];
								//auto voxel = h_cache[index];

								//if (voxel.weight > 0.0f)
								if (-0.05f < voxel.tsdfValue && voxel.tsdfValue < 0.05f)
								{
									//Eigen::Vector3f position = Eigen::Vector3f(
									//	x * cache.voxelSize + cache.cacheMin.x(),
									//	y * cache.voxelSize + cache.cacheMin.y(),
									//	z * cache.voxelSize + cache.cacheMin.z());

									//auto key = GetKey(position);
									auto key = VoxelKey{ x + cacheMinKey.x, y + cacheMinKey.y, z + cacheMinKey.z };
									voxels[key].tsdf = voxel.tsdfValue;
									voxels[key].weight = voxel.weight;
								}
							}
						}
					}
				}

				//cache.Serialize(h_cache.data());
			}

			cudaFree(points);
			cudaDeviceSynchronize();

			for (auto& kvp : voxels)
			{
				auto position = GetPosition(kvp.first);
				if (-0.005f < kvp.second.tsdf && kvp.second.tsdf < 0.005f)
				{
					VisualDebugging::AddCube("Voxel", position, { 0.1f, 0.1f, 0.1f }, { 0.0f, 0.0f, 1.0f }, Color4::White);
				}
			}


			//for (int z = 0; z < cache.cacheSize.z; z++)
			//{
			//	for (int y = 0; y < cache.cacheSize.y; y++)
			//	{
			//		for (int x = 0; x < cache.cacheSize.x; x++)
			//		{
			//			int index = z * cache.cacheSize.z * cache.cacheSize.y + y * cache.cacheSize.x + x;
			//			//auto voxel = cache.cache[index];
			//			auto voxel = h_cache[index];

			//			//if (voxel.weight != 0.0f)
			//			if (-0.05f < voxel.tsdfValue && voxel.tsdfValue < 0.05f)
			//			{
			//				float px = cache.cacheMin.x() + x * cache.voxelSize;
			//				float py = cache.cacheMin.y() + y * cache.voxelSize;
			//				float pz = cache.cacheMin.z() + z * cache.voxelSize;

			//				VisualDebugging::AddCube("Voxel", { px, py, pz }, { 0.1f, 0.1f, 0.1f }, { 0.0f, 0.0f, 1.0f }, Color4::White);
			//			}
			//		}
			//	}
			//}








			//{
			//	float sampleSize = 0.1f;

			//	printf("[[[ patchPoints.size() ]]] : %llu\n", patchPoints.size());

			//	auto t = Time::Now();

			//	pp.DownSample(patchPoints.data(), patchPoints.size(), sampleSize, taabb.min(), taabb.max());

			//	t = Time::End(t, "DownSampling");

			//	printf("[[[ *pp.h_numberOfResultPoints ]]] : %llu\n", pp.h_numberOfResultPoints);

			//	//for (size_t i = 0; i < pp.h_numberOfResultPoints; i++)
			//	//{
			//	//	auto& p = pp.h_resultPoints[i];

			//	//	//p += Eigen::Vector3f(0.0f, 0.0f, 10.0f);

			//	//	VisualDebugging::AddSphere("DownSample", p, { 0.05f, 0.05f, 0.05f }, { 0.0f, 0.0f, 0.0f }, Color4::Blue);
			//	//}
			//}

			//{
			//	float sampleSize = 0.1f;

			//	printf("[[[ patchPoints.size() ]]] : %llu\n", patchPoints.size());

			//	auto t = Time::Now();

			//	pp.MedianFilter(patchPoints.data(), patchPoints.size(), sampleSize, taabb.min(), taabb.max());

			//	t = Time::End(t, "MedianFilter");

			//	printf("[[[ *pp.h_numberOfResultPoints ]]] : %llu\n", pp.h_numberOfResultPoints);

			//	//for (size_t i = 0; i < pp.h_numberOfResultPoints; i++)
			//	//{
			//	//	auto& p = pp.h_resultPoints[i];

			//	//	//p += Eigen::Vector3f(0.0f, 0.0f, 10.0f);

			//	//	VisualDebugging::AddSphere("Filter", p, { 0.05f, 0.05f, 0.05f }, { 0.0f, 0.0f, 0.0f }, Color4::Red);
			//	//}
			//}
		}

		return;

		{
			auto t = Time::Now();

			auto op = Eigen::Vector3f(transform_0[3], transform_0[7], transform_0[11]);

			for (size_t i = 0; i < inputPoints.size(); i++)
			{
				auto& p = inputPoints[i];
				auto key = GetKey(p);
				auto& voxel = voxels[key];

				auto vp = GetPosition(key);
				float tsdf = (op - vp).norm() - (op - p).norm();

				if (0.0f == voxel.weight)
				{
					voxel.tsdf = tsdf;
					voxel.weight = 1.0f;
				}
				else
				{
					float total_weight = voxel.weight + 1.0f;
					voxel.tsdf = (voxel.weight * voxel.tsdf + tsdf) / total_weight;
					voxel.weight = std::min(total_weight, voxel.weight);
				}

				{
					auto p = inputPoints[i] + Eigen::Vector3f(0.1f, 0.0f, 0.0f);
					auto nkey = key;
					nkey.x += 1;
					auto& vVoxel = voxels[nkey];
					auto vp = GetPosition(nkey);
					float tsdf = (op - vp).norm() - (op - p).norm();

					if (0.0f == vVoxel.weight)
					{
						vVoxel.tsdf = tsdf;
						vVoxel.weight = 1.0f;
					}
					else
					{
						float total_weight = vVoxel.weight + 1.0f;
						vVoxel.tsdf = (vVoxel.weight * vVoxel.tsdf + tsdf) / total_weight;
						vVoxel.weight = std::min(total_weight, vVoxel.weight);
					}
				}

				{
					auto p = inputPoints[i] - Eigen::Vector3f(0.1f, 0.0f, 0.0f);
					auto nkey = key;
					nkey.x -= 1;
					auto& vVoxel = voxels[nkey];
					auto vp = GetPosition(nkey);
					float tsdf = (op - vp).norm() - (op - p).norm();

					if (0.0f == vVoxel.weight)
					{
						vVoxel.tsdf = tsdf;
						vVoxel.weight = 1.0f;
					}
					else
					{
						float total_weight = vVoxel.weight + 1.0f;
						vVoxel.tsdf = (vVoxel.weight * vVoxel.tsdf + tsdf) / total_weight;
						vVoxel.weight = std::min(total_weight, vVoxel.weight);
					}
				}

				{
					auto p = inputPoints[i] + Eigen::Vector3f(0.0f, 0.1f, 0.0f);
					auto nkey = key;
					nkey.y += 1;
					auto& vVoxel = voxels[nkey];
					auto vp = GetPosition(nkey);
					float tsdf = (op - vp).norm() - (op - p).norm();

					if (0.0f == vVoxel.weight)
					{
						vVoxel.tsdf = tsdf;
						vVoxel.weight = 1.0f;
					}
					else
					{
						float total_weight = vVoxel.weight + 1.0f;
						vVoxel.tsdf = (vVoxel.weight * vVoxel.tsdf + tsdf) / total_weight;
						vVoxel.weight = std::min(total_weight, vVoxel.weight);
					}
				}

				{
					auto p = inputPoints[i] - Eigen::Vector3f(0.0f, 0.1f, 0.0f);
					auto nkey = key;
					nkey.y -= 1;
					auto& vVoxel = voxels[nkey];
					auto vp = GetPosition(nkey);
					float tsdf = (op - vp).norm() - (op - p).norm();

					if (0.0f == vVoxel.weight)
					{
						vVoxel.tsdf = tsdf;
						vVoxel.weight = 1.0f;
					}
					else
					{
						float total_weight = vVoxel.weight + 1.0f;
						vVoxel.tsdf = (vVoxel.weight * vVoxel.tsdf + tsdf) / total_weight;
						vVoxel.weight = std::min(total_weight, vVoxel.weight);
					}
				}

				{
					auto p = inputPoints[i] + Eigen::Vector3f(0.0f, 0.0f, 0.1f);
					auto nkey = key;
					nkey.z += 1;
					auto& vVoxel = voxels[nkey];
					auto vp = GetPosition(nkey);
					float tsdf = (op - vp).norm() - (op - p).norm();

					if (0.0f == vVoxel.weight)
					{
						vVoxel.tsdf = tsdf;
						vVoxel.weight = 1.0f;
					}
					else
					{
						float total_weight = vVoxel.weight + 1.0f;
						vVoxel.tsdf = (vVoxel.weight * vVoxel.tsdf + tsdf) / total_weight;
						vVoxel.weight = std::min(total_weight, vVoxel.weight);
					}
				}

				{
					auto p = inputPoints[i] - Eigen::Vector3f(0.0f, 0.0f, 0.1f);
					auto nkey = key;
					nkey.z -= 1;
					auto& vVoxel = voxels[nkey];
					auto vp = GetPosition(nkey);
					float tsdf = (op - vp).norm() - (op - p).norm();

					if (0.0f == vVoxel.weight)
					{
						vVoxel.tsdf = tsdf;
						vVoxel.weight = 1.0f;
					}
					else
					{
						float total_weight = vVoxel.weight + 1.0f;
						vVoxel.tsdf = (vVoxel.weight * vVoxel.tsdf + tsdf) / total_weight;
						vVoxel.weight = std::min(total_weight, vVoxel.weight);
					}
				}
			}

			t = Time::End(t, "Voxel Integrate.");

			for (auto& kvp : voxels)
			{
				if (kvp.second.weight > 0.0f)
				{
					if ((-0.5f <= kvp.second.tsdf) && (0.5f >= kvp.second.tsdf))
					{
						auto p = GetPosition(kvp.first);

						VisualDebugging::AddCube("Voxel", p, { 0.1f, 0.1f, 0.1f }, { 0.0f, 0.0f, 1.0f }, Color4::White);
					}
				}
			}
		}

		VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 100.0f, 0.0f, 0.0f }, Color4::Red);
		VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 100.0f, 0.0f }, Color4::Green);
		VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, 100.0f }, Color4::Blue);
		});

	app.AddKeyPressCallback(OnKeyPress);
	app.Run();

	return 0;
}
