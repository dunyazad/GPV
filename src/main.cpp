#include <Common.h>

#include <openvdb/openvdb.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include <App/CustomTrackballStyle.h>
#include <App/Utility.h>

#include <Algorithm/vtkMedianFilter.h>
#include <Algorithm/vtkQuantizingFilter.h>

#include <Debugging/VisualDebugging.h>

void OnKeyPress(vtkObject* caller, long unsigned int eventId, void* clientData, void* callData) {
    vtkRenderWindowInteractor* interactor = static_cast<vtkRenderWindowInteractor*>(caller);
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
	else if (key == "space")
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
	else if (key == "1")
	{
		VisualDebugging::ToggleVisibility("Spheres");
	}
}

class TimerCallback : public vtkCommand
{
public:
    static TimerCallback* New() { return new TimerCallback; }

    TimerCallback() = default;

    virtual void Execute(vtkObject* caller, unsigned long eventId, void* vtkNotUsed(callData)) override
        //virtual void Execute(vtkObject *vtkNotUsed(caller),
        //                     unsigned long vtkNotUsed(eventId),
        //                     void *vtkNotUsed(callData)) override
    {
        if (eventId == vtkCommand::TimerEvent) {
            animate();
        }
        else {
            std::cerr << "Unexpected event ID: " << eventId << std::endl;
        }
    }

private:
    void animate() { VisualDebugging::Update(); }
};

#pragma region KDTree
class KDTreeNode
{
public:
	KDTreeNode(unsigned int point_index) : point_index(point_index) {}

	inline unsigned int GetPointIndex() const { return point_index; }

private:
	unsigned int point_index = UINT_MAX;
	KDTreeNode* left = nullptr;
	KDTreeNode* right = nullptr;

public:
	friend class KDTree;
};

class KDTree
{
public:
	KDTree() : points(nullptr) {}
	KDTree(float* points) : points(points) {}
	~KDTree() { Clear(); }

	void Clear()
	{
		if (nullptr != root)
		{
			ClearRecursive(root);
			root = nullptr;
		}

		nearestNeighborNode = nullptr;
		nearestNeighbor = UINT_MAX;
		nearestNeighborDistance = FLT_MAX;
	}

	void Insert(unsigned int point_index)
	{
		root = InsertRecursive(root, point_index, 0);
	}

	unsigned int FindNearestNeighbor(const float* query)
	{
		/*			nearestNeighbor = root->point_index;

					float* point = GetMappedPoint(root->point_index);

					auto dx = point[0] - query[0];
					auto dy = point[1] - query[1];
					auto dz = point[2] - query[2];
					nearestNeighborDistance = sqrtf(dx * dx + dy * dy + dz * dz);
					nearestNeighborNode = root;
					FindNearestNeighborRecursive(root, query, 0);
					return nearestNeighbor;*/

		nearestNeighbor = UINT_MAX;
		nearestNeighborDistance = FLT_MAX;
		nearestNeighborNode = root;
		FindNearestNeighborRecursive(root, query, 0);
		return nearestNeighborNode->point_index;
	}

	KDTreeNode* FindNearestNeighborNode(const float* query)
	{
		if (nullptr == root)
			return nullptr;

		nearestNeighbor = UINT_MAX;
		nearestNeighborDistance = FLT_MAX;
		nearestNeighborNode = root;
		FindNearestNeighborRecursive(root, query, 0);
		return nearestNeighborNode;
	}

	vector<unsigned int> RangeSearchSquaredDistance(const float* query, float squaredRadius)
	{
		vector<unsigned int> result;
		RangeSearchRecursive(root, query, squaredRadius, result, 0);
		return result;
	}

	inline bool IsEmpty() const { return nullptr == root; }

	inline KDTreeNode* GetRootNode() { return root; }

	inline float* GetPoint(unsigned int index)
	{
		if (nullptr == points) return nullptr;
		else
		{
			if (numberOfPoints <= index) return nullptr;
			else
			{
				return points + index * 3;
			}
		}
	}

	inline float* GetMappedPoint(unsigned int index)
	{
		if (nullptr == points) return nullptr;
		else
		{
			if (indexMapping.size() <= index) return nullptr;
			else
			{
				return points + indexMapping[index] * 3;
			}
		}
	}

	inline vector<unsigned int>& GetIndexMapping() { return indexMapping; }

	inline void SetPoints(float* points, unsigned int nop)
	{
		this->points = points;
		numberOfPoints = nop;
		for (unsigned int i = 0; i < nop; i++)
		{
			indexMapping.push_back(i);
		}
	}

	void Build()
	{
		if (0 != numberOfPoints)
		{
			root = BuildKDTree(0, numberOfPoints, 0);
		}
	}

	void Traverse(KDTreeNode* node, function<void(KDTreeNode*)> callback)
	{
		if (nullptr != node)
		{
			callback(node);

			if (nullptr != node->left) Traverse(node->left, callback);
			if (nullptr != node->right) Traverse(node->right, callback);
		}
	}

private:
	float* points;
	vector<unsigned int> indexMapping;
	unsigned int numberOfPoints = 0;

	KDTreeNode* root = nullptr;
	mutable KDTreeNode* nearestNeighborNode = nullptr;
	mutable unsigned int nearestNeighbor = UINT_MAX;
	mutable float nearestNeighborDistance = FLT_MAX;

	bool comparePoints(unsigned int i0, unsigned int i1, unsigned int dim) {
		return GetMappedPoint(i0)[dim] < GetMappedPoint(i1)[dim];
	}

	KDTreeNode* BuildKDTree(unsigned int start, unsigned int end, unsigned int depth)
	{
		if (start >= end) {
			return nullptr;
		}

		unsigned int dim = depth % 3;

		std::sort(indexMapping.begin() + start, indexMapping.begin() + end,
			[&](unsigned int a, unsigned int b) { return comparePoints(a, b, dim); });

		unsigned int median = (start + end) / 2;
		KDTreeNode* node = new KDTreeNode(indexMapping[median]);

		node->left = BuildKDTree(start, median, depth + 1);
		node->right = BuildKDTree(median + 1, end, depth + 1);

		return node;
	}

	void ClearRecursive(KDTreeNode* node)
	{
		if (nullptr != node->left)
		{
			ClearRecursive(node->left);
		}

		if (nullptr != node->right)
		{
			ClearRecursive(node->right);
		}

		delete node;
	}

	KDTreeNode* InsertRecursive(KDTreeNode* node, unsigned int point_index, int depth)
	{
		if (node == nullptr)
		{
			auto newNode = new KDTreeNode(point_index);
			return newNode;
		}

		int currentDimension = depth % 3;
		float pointValue = GetMappedPoint(point_index)[currentDimension];
		float nodeValue = GetMappedPoint(node->point_index)[currentDimension];

		if (pointValue < nodeValue)
		{
			node->left = InsertRecursive(node->left, point_index, depth + 1);
		}
		else
		{
			node->right = InsertRecursive(node->right, point_index, depth + 1);
		}

		return node;
	}

	void FindNearestNeighborRecursive(KDTreeNode* node, const float* query, int depth)
	{
		if (node == nullptr)
		{
			return;
		}

		int currentDimension = depth % 3;

		float* point = GetMappedPoint(node->point_index);

		auto dx = point[0] - query[0];
		auto dy = point[1] - query[1];
		auto dz = point[2] - query[2];
		float nodeDistance = sqrtf(dx * dx + dy * dy + dz * dz);
		if (nodeDistance < nearestNeighborDistance)
		{
			nearestNeighborNode = node;
			nearestNeighbor = node->point_index;
			nearestNeighborDistance = nodeDistance;
		}

		auto queryValue = query[currentDimension];
		auto nodeValue = GetMappedPoint(node->point_index)[currentDimension];

		KDTreeNode* closerNode = nullptr;
		KDTreeNode* otherNode = nullptr;

		if (queryValue < nodeValue)
		{
			closerNode = node->left;
			otherNode = node->right;
		}
		else
		{
			closerNode = node->right;
			otherNode = node->left;
		}

		FindNearestNeighborRecursive(closerNode, query, depth + 1);

		float planeDistance = queryValue - nodeValue;
		if (planeDistance * planeDistance < nearestNeighborDistance) {
			FindNearestNeighborRecursive(otherNode, query, depth + 1);
		}
	}

	void RangeSearchRecursive(KDTreeNode* node, const float* query, float squaredRadius, std::vector<unsigned int>& result, int depth)
	{
		if (node == nullptr)
		{
			return;
		}

		float* point = GetMappedPoint(node->point_index);

		auto dx = point[0] - query[0];
		auto dy = point[1] - query[1];
		auto dz = point[2] - query[2];
		float nodeDistance = sqrtf(dx * dx + dy * dy + dz * dz);

		if (nodeDistance <= squaredRadius)
		{
			result.push_back(node->point_index);
		}

		int currentDimension = depth % 3;
		float queryValue = query[currentDimension];
		float nodeValue = GetMappedPoint(node->point_index)[currentDimension];

		KDTreeNode* closerNode = (queryValue < nodeValue) ? node->left : node->right;
		KDTreeNode* otherNode = (queryValue < nodeValue) ? node->right : node->left;

		RangeSearchRecursive(closerNode, query, squaredRadius, result, depth + 1);

		float distanceToPlane = queryValue - nodeValue;
		if (distanceToPlane * distanceToPlane <= squaredRadius)
		{
			RangeSearchRecursive(otherNode, query, squaredRadius, result, depth + 1);
		}
	}
};

float GetDistanceSquared(const float* points, unsigned int point_index, const float* query)
{
	float distanceSquared = 0.0f;
	for (int i = 0; i < 3; ++i) {
		float diff = points[point_index * 3 + i] - query[i];
		distanceSquared += diff * diff;
	}
	return distanceSquared;
}

float GetDistance(const float* points, unsigned int point_index, const float* query)
{
	auto dx = points[point_index * 3] - query[0];
	auto dy = points[point_index * 3 + 1] - query[1];
	auto dz = points[point_index * 3 + 2] - query[2];
	return sqrtf(dx * dx + dy * dy + dz * dz);
}
#pragma endregion

int main() {
    openvdb::initialize();

    MaximizeConsoleWindowOnMonitor(1);

    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->SetBackground(0.3, 0.5, 0.7);

    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->SetSize(1920, 1080);
    renderWindow->AddRenderer(renderer);

    vtkSmartPointer<vtkRenderWindowInteractor> interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    //vtkNew<vtkInteractorStyleTrackballCamera> trackballStyle;
    //interactor->SetInteractorStyle(trackballStyle);
    vtkNew<CustomTrackballStyle> customTrackballStyle;
    interactor->SetInteractorStyle(customTrackballStyle);
    interactor->SetRenderWindow(renderWindow);
    interactor->Initialize();

    VisualDebugging::Initialize(renderer);

    MaximizeVTKWindowOnMonitor(renderWindow, 2);

    auto inputPoints = ReadPLY("C:\\Resources\\Debug\\patches\\0.ply");

	{
		int K = 20;

		vtkSmartPointer<vtkKdTree> kdTree = vtkSmartPointer<vtkKdTree>::New();
		kdTree->BuildLocatorFromPoints(inputPoints);

		vtkSmartPointer<vtkPoints> filteredPoints = vtkSmartPointer<vtkPoints>::New();
		filteredPoints->SetNumberOfPoints(inputPoints->GetNumberOfPoints());

		// Iterate over all points in the cloud
		for (vtkIdType i = 0; i < inputPoints->GetNumberOfPoints(); ++i)
		{
			double point[3];
			inputPoints->GetPoint(i, point);

			// Find the K-nearest neighbors
			vtkSmartPointer<vtkIdList> result = vtkSmartPointer<vtkIdList>::New();
			kdTree->FindClosestNPoints(K, point, result);

			// Store the coordinates of the neighbors
			std::vector<double> x_vals, y_vals, z_vals;
			for (vtkIdType j = 0; j < result->GetNumberOfIds(); ++j)
			{
				double neighbor[3];
				inputPoints->GetPoint(result->GetId(j), neighbor);
				x_vals.push_back(neighbor[0]);
				y_vals.push_back(neighbor[1]);
				z_vals.push_back(neighbor[2]);
			}

			// Compute the median for each coordinate
			std::sort(x_vals.begin(), x_vals.end());
			std::sort(y_vals.begin(), y_vals.end());
			std::sort(z_vals.begin(), z_vals.end());

			double median_point[3];
			median_point[0] = x_vals[K / 2];
			median_point[1] = y_vals[K / 2];
			median_point[2] = z_vals[K / 2];

			// Set the new point into the filtered point cloud
			filteredPoints->SetPoint(i, median_point);

			VisualDebugging::AddSphere(
				"Spheres",
				{ (float)median_point[0], (float)median_point[1], (float)median_point[2] },
				{ 0.1f, 0.1f, 0.1f },
				{ 0, 0, 0 },
				255, 255, 255);
		}
	}

 //   vtkSmartPointer<vtkQuantizingFilter> quantizingFilter = vtkSmartPointer<vtkQuantizingFilter>::New();
	//quantizingFilter->SetInputData(inputPoints);
	//quantizingFilter->Update();

    //auto inputPoints = ReadPLY("C:\\Resources\\Debug\\ZeroCrossingPoints.ply");
  //  {
  //      vtkSmartPointer<vtkVertexGlyphFilter> vertexFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
  //      vertexFilter->SetInputData(inputPoints);
  //      vertexFilter->Update();

  //      vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  //      mapper->SetInputConnection(vertexFilter->GetOutputPort());

  //      vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
  //      actor->SetMapper(mapper);
  //      actor->GetProperty()->SetColor(1.0, 0.0, 0.0);
		//actor->GetProperty()->SetPointSize(3);

  //      renderer->AddActor(actor);
  //  }

    auto bounds = inputPoints->GetBounds();

    VisualDebugging::AddLine("axes", { 0, 0, 0 }, { (float)bounds[1], 0, 0 }, 255, 0, 0);
    VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0, (float)bounds[3], 0 }, 0, 255, 0);
    VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0, 0, (float)bounds[5] }, 0, 0, 255);

	{
		float x = (float)bounds[0];
		float X = (float)bounds[1];
		float y = (float)bounds[2];
		float Y = (float)bounds[3];
		float z = (float)bounds[4];
		float Z = (float)bounds[5];

		VisualDebugging::AddLine("bb", { x, y, z }, { X, y, z }, 255, 255, 255);
		VisualDebugging::AddLine("bb", { x, y, z }, { x, Y, z }, 255, 255, 255);
		VisualDebugging::AddLine("bb", { x, y, z }, { x, y, Z }, 255, 255, 255);

		VisualDebugging::AddLine("bb", { x, Y, Z }, { X, Y, Z }, 255, 255, 255);
		VisualDebugging::AddLine("bb", { X, y, Z }, { X, Y, Z }, 255, 255, 255);
		VisualDebugging::AddLine("bb", { X, Y, z }, { X, Y, Z }, 255, 255, 255);
	}

	{
		VisualDebugging::AddLine("Region", { -12.8f, -24.0f, 0.0f }, {  12.8f, -24.0f, 0.0f }, 255, 255, 255);
		VisualDebugging::AddLine("Region", {  12.8f, -24.0f, 0.0f }, {  12.8f,  24.0f, 0.0f }, 255, 255, 255);
		VisualDebugging::AddLine("Region", {  12.8f,  24.0f, 0.0f }, { -12.8f,  24.0f, 0.0f }, 255, 255, 255);
		VisualDebugging::AddLine("Region", { -12.8f,  24.0f, 0.0f }, { -12.8f, -24.0f, 0.0f }, 255, 255, 255);
	}

	//for (size_t w = 0; w < 256; w++)
	//{
	//	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { (float)bounds[1], 0, 0 }, 255, 255, 255);
	//}

	//for (size_t h = 0; h < 480; h++)
	//{
	//}

  //  {
		//auto points = quantizingFilter->GetOutput();
		//auto nop = points->GetNumberOfPoints();
  //      for (size_t i = 0; i < nop; i++)
  //      {
  //          auto p = points->GetPoint(i);
		//	if (-1000 != p[2])
		//	{
		//		VisualDebugging::AddSphere(
		//			"Spheres",
		//			{ (float)p[0], (float)p[1], (float)p[2] },
		//			{ 0.1f, 0.1f, 0.1f },
		//			{ 0, 0, 0 },
		//			255, 255, 255);
		//	}
  //      }
  //  }

    vtkSmartPointer<vtkCallbackCommand> keyPressCallback = vtkSmartPointer<vtkCallbackCommand>::New();
    keyPressCallback->SetCallback(OnKeyPress);
    keyPressCallback->SetClientData(renderer);

    vtkSmartPointer<TimerCallback> timerCallback = vtkSmartPointer<TimerCallback>::New();

    interactor->AddObserver(vtkCommand::TimerEvent, timerCallback);
    int timerId = interactor->CreateRepeatingTimer(16);
    if (timerId < 0) {
        std::cerr << "Error: Timer was not created!" << std::endl;
    }

    interactor->AddObserver(vtkCommand::KeyPressEvent, keyPressCallback);

    renderWindow->Render();
    interactor->Start();

    VisualDebugging::Terminate();

    return 0;
}
