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

#include <CUDA/KDTree.cuh>

int pid = 0;
size_t size_0 = 0;
size_t size_45 = 0;
float transform_0[16];
float transform_45[16];
unsigned char image_0[400 * 480];
unsigned char image_45[400 * 480];
Eigen::Vector3f points_0[400 * 480];
Eigen::Vector3f points_45[400 * 480];

int index = 0;





void Swap(int* arr, int a, int b) {
	int temp = arr[a];
	arr[a] = arr[b];
	arr[b] = temp;
}

class BitonicSorter {
public:
	bool UP = true, DOWN = false; //오름차순, 내림차순 (for Bitonic Sort)

	void BitonicSort(int* arr, int start, int n, bool direction) { //start = 0, dir = 1
		if (n > 1) { //comparator 가 비교할수있는 최소 데이터 갯수는 2개이다. 
			int k = n / 2; //계속 반으로 줄여 나간다. 
			//Conquer 
			//comparator를 이요해 bitonic sequence를 만들기 위해 데이터를 반으로 나눈다. 
			BitonicSort(arr, start, k, UP); //반은 오름차순
			BitonicSort(arr, start + k, k, DOWN); //다음 반은 내림차순 
			//Combine
			BitonicMerge(arr, start, n, direction); //오름 차순 혹은 내림차순으로 전체 정렬을 수행한다. 
		}
	}
	void BitonicMerge(int* arr, int start, int n, bool direction) {
		if (n > 1) {//comparator 가 비교할수있는 최소 데이터 갯수는 2개이다. 
			int k = n / 2;
			//반으로 나눠진 데이터는 comparator을 지나 각각 bitonic sequence 를 가지게 된다.
			for (int i = start; i < start + k; i++) Comparator(arr, i, i + k, direction);
			//Conquer
			BitonicMerge(arr, start, k, direction); //반씩 나눠가며 재정렬
			BitonicMerge(arr, start + k, k, direction);//반씩 나눠가며 재정렬
			//Combine
		}
	}
	void Comparator(int* arr, int i, int j, bool direction) {
		//dir 이 오름 차순일때, 내림차순일때에 따라 정렬한다. 
		//오름차순(1)일 때, 앞>뒤 true 이면 swap
		//내림차순(0)일 때, 앞>뒤 false 이면 swap
		if (direction == (arr[i] > arr[j])) Swap(arr, i, j);
	}
};

//void Bitonic(int* arr, int n) { //Bitonic Sort
//	BitonicSorter bs;
//	chrono::steady_clock::time_point start, end;
//	chrono::nanoseconds result;
//	start = chrono::steady_clock::now();
//
//	bs.BitonicSort(arr, 0, n, true); //오름 차순 정렬 ( true )
//
//	end = chrono::steady_clock::now(); result = end - start;
//	cout << "4. Bitonic Sort                : " << result.count() << "ns" << endl; //알고리즘 실행시간
//}

bool isPowerOfTwo(int n) {
	return (n && !(n & (n - 1)));
}

int nextPowerOfTwo(int n) {
	return pow(2, ceil(log2(n)));
}

void Bitonic(int* arr, int n) {
	// Step 1: Find next power of two
	int nextPOT = nextPowerOfTwo(n);

	// Step 2: Create a new array of size nextPOT
	int* paddedArr = new int[nextPOT];

	// Copy the original array and pad with INT_MAX (for ascending sort)
	for (int i = 0; i < n; i++) paddedArr[i] = arr[i];
	for (int i = n; i < nextPOT; i++) paddedArr[i] = INT_MAX; // Pad with max value

	// Step 3: Perform Bitonic Sort on the padded array
	BitonicSorter bs;
	chrono::steady_clock::time_point start, end;
	chrono::nanoseconds result;
	start = chrono::steady_clock::now();

	bs.BitonicSort(paddedArr, 0, nextPOT, true); // 오름차순 정렬 (true)

	end = chrono::steady_clock::now();
	result = end - start;
	cout << "4. Bitonic Sort                : " << result.count() << "ns" << endl; // 알고리즘 실행시간

	// Step 4: Copy back only the first 'n' sorted elements to the original array
	for (int i = 0; i < n; i++) arr[i] = paddedArr[i];

	// Cleanup
	delete[] paddedArr;
}

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
					VisualDebugging::AddSphere("Depth", { (float)x / (float)width * 100.0f, (float)y / (float)height * 100.0f, normalizedDepth * 100.0f }, { 0.1f, 0.1f, 0.1f }, { 0.0f, 0.0f, 0.0f }, 255, 255, 255);
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
		VisualDebugging::ToggleVisibility("Spheres");
	}
	else if (key == "Right")
	{
	/*	for (size_t i = 0; i < 10; i++)
		{
			int h = index / 256;
			int w = index % 256;

			float x = (float)(w - 128) * 0.01f;
			float y = (float)(h - 80) * 0.01f;

			auto& p = points_0[(h * 3) * 256 + 255 - w] * 0.1f;

			VisualDebugging::AddSphere("patch", { x, y, p.z() }, { 0.01f, 0.01f, 0.01f }, { 0.0f, 0.0f, 0.0f }, 0, 0, 255);

			index++;
		}*/

		for (size_t i = 0; i < 100; i++)
		{
			Eigen::Vector3f p = points_0[index];
			while (p.x() == FLT_MAX || p.y() == FLT_MAX || p.z() == FLT_MAX)
			{
				index++;
				p = points_0[index];
			}
			VisualDebugging::AddSphere("patch", p, { 0.1f, 0.1f, 0.1f }, { 0.0f, 0.0f, 0.0f }, 0, 0, 255);

			printf("%f, %f, %f\n", p.x(), p.y(), p.z());

			index++;
		}
	}
}

class TimerCallback : public vtkCommand
{
public:
    static TimerCallback* New() { return new TimerCallback; }

    TimerCallback() = default;

    virtual void Execute(vtkObject* caller, unsigned long eventId, void* vtkNotUsed(callData)) override
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

// Bilateral filter function for 3D point cloud
vtkSmartPointer<vtkPoints> BilateralFilter(vtkSmartPointer<vtkPolyData> inputCloud,
	double spatialSigma, double featureSigma,
	double neighborhoodSize)
{
	// Create output cloud
	vtkSmartPointer<vtkPoints> outputPoints = vtkSmartPointer<vtkPoints>::New();
	outputPoints->SetNumberOfPoints(inputCloud->GetNumberOfPoints());

	// Use a KdTree for neighborhood search
	vtkSmartPointer<vtkKdTreePointLocator> kdTree = vtkSmartPointer<vtkKdTreePointLocator>::New();
	kdTree->SetDataSet(inputCloud);
	kdTree->BuildLocator();

	// Iterate over each point in the point cloud
	for (vtkIdType i = 0; i < inputCloud->GetNumberOfPoints(); i++)
	{
		double p[3];
		inputCloud->GetPoint(i, p);
		if (p[0] == FLT_MAX || p[1] == FLT_MAX || p[2] == FLT_MAX)
			continue;
		if (p[0] < -1000 || p[1] < -1000 || p[2] < -1000)
			continue;
		if (p[0] > 1000 || p[1] > 1000 || p[2] > 1000)
			continue;

		// Find the neighboring points using KdTree within a radius
		vtkSmartPointer<vtkIdList> neighbors = vtkSmartPointer<vtkIdList>::New();
		kdTree->FindPointsWithinRadius(neighborhoodSize, p, neighbors);

		// Bilateral filter weights and summation variables
		double sumWeights = 0.0;
		double newPoint[3] = { 0.0, 0.0, 0.0 };

		// Iterate over neighbors
		auto noi = neighbors->GetNumberOfIds();
		for (vtkIdType j = 0; j < noi; j++)
		{
			vtkIdType neighborId = neighbors->GetId(j);
			double neighborPoint[3];
			inputCloud->GetPoint(neighborId, neighborPoint);

			// Calculate spatial distance
			double spatialDist = vtkMath::Distance2BetweenPoints(p, neighborPoint);
			double spatialWeight = exp(-spatialDist / (2.0 * spatialSigma * spatialSigma));

			// Calculate feature distance (here we use distance itself, can be curvature or normal)
			double featureDist = vtkMath::Distance2BetweenPoints(p, neighborPoint);
			double featureWeight = exp(-featureDist / (2.0 * featureSigma * featureSigma));

			// Total weight
			double totalWeight = spatialWeight * featureWeight;

			// Update the weighted sum for the new point position
			newPoint[0] += totalWeight * neighborPoint[0];
			newPoint[1] += totalWeight * neighborPoint[1];
			newPoint[2] += totalWeight * neighborPoint[2];

			sumWeights += totalWeight;
		}

		// Normalize the new point
		newPoint[0] /= sumWeights;
		newPoint[1] /= sumWeights;
		newPoint[2] /= sumWeights;

		// Set the new filtered point
		outputPoints->SetPoint(i, newPoint);

		VisualDebugging::AddSphere("Spheres", { (float)newPoint[0],(float)newPoint[1],(float)newPoint[2] },
			{ 0.05f, 0.05f, 0.05f }, { 0, 0, 0 }, 255, 255, 255);
	}

	// Update the input point cloud with the smoothed points
	//inputCloud->SetPoints(outputPoints);

	return outputPoints;
}

vtkSmartPointer<vtkPolyData> CreatePatchMesh(int width, int height)
{
	vtkNew<vtkPoints> points;
	for (int h = 0; h < height; h++)
	{
		for (int w = 0; w < width; w++)
		{
			float x = ((float)w - (float)width * 0.5f) * 0.1f;
			float y = ((float)h - (float)height * 0.5f) * 0.1f;
			float z = 0.0f;

			auto& p = points_0[h * width + w];

			if ((p.x() > -1000 && p.y() > -1000 && p.z() > -1000) &&
				(p.x() < 1000 || p.y() < 1000 || p.z() < 1000))
			{
				z = p.z();
			}
			
			points->InsertNextPoint(x, y, z);
		}
	}

	vtkNew<vtkCellArray> triangles;

	for (int h = 0; h < height - 1; h++)
	{
		for (int w = 0; w < width - 1; w++)
		{
			vtkIdType p1 = h * width + w;
			vtkIdType p2 = p1 + 1;
			vtkIdType p3 = (h + 1) * width + w + 1;
			vtkIdType p4 = (h + 1) * width + w;

			vtkNew<vtkTriangle> triangle1;
			triangle1->GetPointIds()->SetId(0, p1);
			triangle1->GetPointIds()->SetId(1, p2);
			triangle1->GetPointIds()->SetId(2, p4);
			triangles->InsertNextCell(triangle1);

			vtkNew<vtkTriangle> triangle2;
			triangle2->GetPointIds()->SetId(0, p2);
			triangle2->GetPointIds()->SetId(1, p3);
			triangle2->GetPointIds()->SetId(2, p4);
			triangles->InsertNextCell(triangle2);
		}
	}

	vtkNew<vtkPolyData> polyData;
	polyData->SetPoints(points);
	polyData->SetPolys(triangles);

	return polyData;
}

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

	int count = 0;
	for (size_t i = 0; i < size_0; i++)
	{
		auto& p = points_0[i];

		if ((p.x() > -1000 && p.y() > -1000 && p.z() > -1000) &&
			(p.x() < 1000 && p.y() < 1000 && p.z() < 1000))
		{
			points->InsertNextPoint(p.data());

			count++;
		}
	}

	vtkNew<vtkPolyData> polyData;
	polyData->SetPoints(points);

	WritePLY(polyData, "C:\\Debug\\GPV\\Original.ply");

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
	actor->SetMapper(mapper);

	actor->GetProperty()->SetPointSize(5.0f);
	actor->GetProperty()->SetColor(1.0, 0.0, 0.0);

	renderer->AddActor(actor);
}

Eigen::Vector3f FindMedian(int cx, int cy, int nnn, int fallbackCount = 100)
{
	int currentOffset = 0;
	int found = 0;
	fallbackCount += nnn;
	vector<float> foundX;
	vector<float> foundY;
	vector<float> foundZ;
	//while (currentOffset <= offset)
	while (found <= nnn)
	{
		fallbackCount--;
		if (fallbackCount < 0) break;

		for (int y = -currentOffset; y <= currentOffset; y++)
		{
			if (cy + y < 0) continue;

			for (int x = -currentOffset; x <= currentOffset; x++)
			{
				if (cx + x < 0) continue;

				if ((x == -currentOffset || x == currentOffset) ||
					(y == -currentOffset || y == currentOffset))
				{
					//printf("%d, %d, \n", cx + x, cy + y);

					//VisualDebugging::AddSphere("Spheres", { (float)(cx + x), (float)(cy + y), 0.0f }, { 0.1f, 0.1f, 0.1f }, { 0.0f, 0.0f, 0.0f }, 255, 0, 0);

					auto& p = points_0[(cy + y) * 256 + (cx + x)];
					
					if (FLT_MAX == p.x() || FLT_MAX == p.y() || FLT_MAX == p.z()) continue;

					//VisualDebugging::AddSphere("Filtered Spheres", p, { 0.1f, 0.1f, 0.1f }, { 0.0f, 0.0f, 0.0f }, 255, 0, 0);

					foundX.push_back(p.x());
					foundY.push_back(p.y());
					foundZ.push_back(p.z());
					
					found++;
					if (found >= nnn) break;
				}

				if (found >= nnn) break;
			}

			if (found >= nnn) break;
		}
		currentOffset++;

		if (found >= nnn) break;
	}

	//sort(foundX.begin(), foundX.end());
	//sort(foundY.begin(), foundY.end());
	//sort(foundZ.begin(), foundZ.end());

	if (found == 0)
		return { FLT_MAX, FLT_MAX, FLT_MAX };

	else if (found == 1)
		return { foundX[0], foundY[0], foundZ[0] };
	else if (found % 2 == 0)
	{
		float x = (foundX[found / 2] + foundX[found / 2 + 1]) * 0.5f;
		float y = (foundY[found / 2] + foundY[found / 2 + 1]) * 0.5f;
		float z = (foundZ[found / 2] + foundZ[found / 2 + 1]) * 0.5f;

		return { x, y, z };
	}
	else
	{
		return { foundX[found / 2 + 1], foundY[found / 2 + 1], foundZ[found / 2 + 1] };
	}
}

void MedianFilter(int nnn)
{
	vector<Eigen::Vector3f> newPoints(256 * 480, Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX));

	for (int h = 0; h < 480; h++)
	{
		for (int w = 0; w < 256; w++)
		{
			newPoints[256 * h + w] = FindMedian(w, h, 3);
		}
	}

	for (size_t i = 0; i < 256 * 480; i++)
	{
		auto& p = newPoints[i];

		if (FLT_MAX == p.x() || FLT_MAX == p.y() || FLT_MAX == p.z()) continue;

		//if (i > 100) return;

		VisualDebugging::AddSphere("Filtered", p, { 0.1f, 0.1f, 0.1f }, { 0.0f, 0.0f, 0.0f }, 0, 0, 255);
	}
}

int main() {
    openvdb::initialize();

    MaximizeConsoleWindowOnMonitor(1);

    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->SetBackground(0.3, 0.5, 0.7);

    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->SetSize(1920, 1080);
	//renderWindow->SetSize(256, 480);
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

	LoadPatch(3, renderer);

	{
		auto polyData = CreatePatchMesh(256, 480);
		//auto polyData = CreatePatchMesh(128, 160);

		vtkNew<vtkPolyDataMapper> mapper;
		mapper->SetInputData(polyData);

		vtkNew<vtkActor> actor;
		actor->SetMapper(mapper);

		renderer->AddActor(actor);
	}

#pragma region CPU Bitonic Sort
	//int arraySize = 256 * 480;

	//std::random_device rd;
	//std::mt19937 gen(rd());
	//// Define a distribution (e.g., numbers between 0 and arraySize)
	//std::uniform_int_distribution<> distrib(0, arraySize);

	//// Declare an array of arraySize
	//std::vector<int> randomArray(arraySize);

	//// Fill the array with random numbers
	//for (int& num : randomArray) {
	//	num = distrib(gen);
	//}

	//for (size_t i = 0; i < randomArray.size(); i++)
	//{
	//	VisualDebugging::AddSphere("Array", { (float)i, (float)randomArray[i], 0.0f }, { (float)arraySize / 1000.0f, (float)arraySize / 1000.0f, (float)arraySize / 1000.0f }, { 0.0f, 0.0f, 0.0f }, 255, 255, 255);
	//}

	//Bitonic(randomArray.data(), randomArray.size());

	//for (size_t i = 0; i < randomArray.size(); i++)
	//{
	//	VisualDebugging::AddSphere("Array", { (float)i, (float)randomArray[i], 0.0f }, { (float)arraySize / 1000.0f, (float)arraySize / 1000.0f, (float)arraySize / 1000.0f }, { 0.0f, 0.0f, 0.0f }, 0, 0, 255);
	//}
#pragma endregion

	//vector<Eigen::Vector3f> result = CUDA::DoFilter((Eigen::Vector3f*)points_0->data());
	//for (auto& p : result)
	//{
	//	VisualDebugging::AddSphere("result", p, { 0.1f, 0.1f, 0.1f }, { 0.0f, 0.0f, 0.0f }, 0, 0, 255);
	//}
	////MedianFilter(30);

	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { (float)128, 0, 0 }, 255, 0, 0);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0, (float)240, 0 }, 0, 255, 0);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0, 0, (float)240 }, 0, 0, 255);

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

	//vtkCamera* camera = renderer->GetActiveCamera();
	//camera->SetParallelProjection(true);
	//renderer->ResetCamera();

	CUDA::Test();

    renderWindow->Render();
    interactor->Start();

    VisualDebugging::Terminate();

    return 0;
}
