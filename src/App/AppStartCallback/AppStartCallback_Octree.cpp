#include <App/AppStartCallback/AppStartCallback.h>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

#define MORTON_CODE(code, depth, maxDepth) ((code >> (3 * (maxDepth - depth))) & 0b111)

struct Octant
{
	size_t code = UINT64_MAX;
	Octant* children[8] = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
};

struct Octree
{
	Octant* root = nullptr;
	Eigen::Vector3f min;
	Eigen::Vector3f max;
};

uint64_t GetMortonCode(const Eigen::Vector3f& max, const Eigen::Vector3f& min, int maxDepth, const Eigen::Vector3f& position)
{
	// Validate and compute range
	Eigen::Vector3f range = max - min;
	range = range.cwiseMax(Eigen::Vector3f::Constant(1e-6f)); // Avoid zero range

	// Normalize position
	Eigen::Vector3f relativePos = (position - min).cwiseQuotient(range);

	// Clamp to [0, 1]
	relativePos = relativePos.cwiseMax(0.0f).cwiseMin(1.0f);

	// Scale to Morton grid size
	uint32_t maxCoordinateValue = (1 << maxDepth) - 1;
	uint32_t x = static_cast<uint32_t>(relativePos.x() * maxCoordinateValue);
	uint32_t y = static_cast<uint32_t>(relativePos.y() * maxCoordinateValue);
	uint32_t z = static_cast<uint32_t>(relativePos.z() * maxCoordinateValue);
	
	// Compute Morton code
	uint64_t mortonCode = 0;
	for (int i = 0; i < maxDepth; ++i) {
		mortonCode |= ((x >> i) & 1ULL) << (3 * i);
		mortonCode |= ((y >> i) & 1ULL) << (3 * i + 1);
		mortonCode |= ((z >> i) & 1ULL) << (3 * i + 2);
	}

	return mortonCode;
}

// Utility Functions for Morton Code
uint32_t ExtractBitsFromMorton(uint64_t mortonCode, int startBit, int depth) {
	uint32_t value = 0;
	for (int i = 0; i < depth; ++i) {
		value |= ((mortonCode >> (3 * i + startBit)) & 1ULL) << i;
	}
	return value;
}

uint32_t ExtractValueFromMorton(uint64_t mortonCode, int startBit, int depth) {
	uint32_t value = 0;
	for (int i = 0; i < depth; ++i) {
		value |= ((mortonCode >> (3 * i + startBit)) & 111ULL) << i;
	}
	return value;
}

Eigen::Vector3f CalculatePositionFromMortonCode(uint64_t mortonCode, int depth, const Eigen::Vector3f& min, const Eigen::Vector3f& max) {
	uint32_t x = ExtractBitsFromMorton(mortonCode, 0, depth);
	uint32_t y = ExtractBitsFromMorton(mortonCode, 1, depth);
	uint32_t z = ExtractBitsFromMorton(mortonCode, 2, depth);

	uint32_t numSubdivisions = 1 << depth;
	Eigen::Vector3f voxelSize = (max - min) / numSubdivisions;

	return min + Eigen::Vector3f(x, y, z).cwiseProduct(voxelSize) + (voxelSize * 0.5f);
}

Eigen::Vector3f CalculateVoxelSizeFromMortonCode(uint64_t mortonCode, int depth, const Eigen::Vector3f& min, const Eigen::Vector3f& max)
{
	Eigen::Vector3f span = max - min;
	float subdivisions = static_cast<float>(1 << depth); // 2^depth
	Eigen::Vector3f voxelSize = span / subdivisions;
	return voxelSize;
}

template<typename T>
void Plot(App* pApp, const vector<T>& values)
{
	vtkSmartPointer<vtkTable> table = vtkSmartPointer<vtkTable>::New();

	vtkSmartPointer<vtkFloatArray> xArr = vtkSmartPointer<vtkFloatArray>::New();
	xArr->SetName("X");
	table->AddColumn(xArr);

	vtkSmartPointer<vtkFloatArray> yArr = vtkSmartPointer<vtkFloatArray>::New();
	yArr->SetName("Y");
	table->AddColumn(yArr);

	int numPoints = 500;
	table->SetNumberOfRows(values.size());
	for (int i = 0; i < values.size(); ++i)
	{
		table->SetValue(i, 0, i);
		table->SetValue(i, 1, values[i]);
	}

	vtkSmartPointer<vtkChartXY> chart = vtkSmartPointer<vtkChartXY>::New();

	vtkPlot* line = chart->AddPlot(vtkChart::LINE);
	line->SetInputData(table, 0, 1); // X column is 0, Y column is 1
	line->SetColor(0, 255, 0, 255); // RGBA color: green
	line->SetWidth(2.0);            // Line width

	vtkSmartPointer<vtkContextActor> contextActor = vtkSmartPointer<vtkContextActor>::New();
	contextActor->GetScene()->AddItem(chart);

	pApp->GetChartRenderer()->AddActor(contextActor);
}

void Populate(Octant* octant, size_t mortonCode, int depth, int maxDepth)
{
	auto index = MORTON_CODE(mortonCode, depth, maxDepth);

	if (nullptr == octant->children[index])
	{
		octant->children[index] = new Octant;
		//octant->children[index]->code = 
	}

	if (depth <= maxDepth)
	{
		Populate(octant->children[index], mortonCode, depth + 1, maxDepth);
	}
}

void PopulateOctree(Octree* octree, vector<size_t>& mortonCodes, int maxDepth)
{
	if (nullptr == octree->root)
	{
		octree->root = new Octant;
	}

	for (size_t i = 0; i < mortonCodes.size(); i++)
	{
		Populate(octree->root, mortonCodes[i], 1, maxDepth);
	}
}

void Visualize(Octant* octant, int depth, int maxDepth, const Eigen::Vector3f& center, float halfSize)
{
	if (depth <= maxDepth)
	{
		stringstream ss;
		ss << "Cubes_" << depth;
		VD::AddCube(ss.str(), center, halfSize, Color4::White);

		for (size_t i = 0; i < 8; i++)
		{
			if (nullptr != octant->children[i])
			{
				auto childCenter = center;
				if (i & 0b001) childCenter.x() += halfSize * 0.5f;
				else childCenter.x() -= halfSize * 0.5f;
				if (i & 0b010) childCenter.y() += halfSize * 0.5f;
				else childCenter.y() -= halfSize * 0.5f;
				if (i & 0b100) childCenter.z() += halfSize * 0.5f;
				else childCenter.z() -= halfSize * 0.5f;

				Visualize(octant->children[i], depth + 1, maxDepth, childCenter, halfSize * 0.5f);
			}
		}
	}
}

void VisualizeOctree(Octree* octree, int maxDepth, const Eigen::Vector3f& center, float halfSize)
{
	if (nullptr != octree->root)
	{
		Visualize(octree->root, 0, maxDepth, center, halfSize);
	}
}

void AppStartCallback_Octree(App* pApp)
{
	int maxDepth = 15;

	//Plot(pApp);
	//return;

	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 100 * 0.5f, 0.0f, 0.0f }, Color4::Red);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 100 * 0.5f, 0.0f }, Color4::Green);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, 100 * 0.5f }, Color4::Blue);

	PLYFormat ply;
	ply.Deserialize("C:\\Resources\\3D\\PLY\\Complete\\Lower_pointcloud.ply");
	auto& aabb = ply.GetAABB();

	auto aabbMin = Eigen::Vector3f(aabb.min().minCoeff(), aabb.min().minCoeff(), aabb.min().minCoeff());
	auto aabbMax = Eigen::Vector3f(aabb.max().maxCoeff(), aabb.max().maxCoeff(), aabb.max().maxCoeff());

	Octant* octants = new Octant[ply.GetPoints().size() / 3];

	vector<size_t> toSort;

	for (size_t i = 0; i < ply.GetPoints().size() / 3; i++)
	{
		auto x = ply.GetPoints()[i * 3];
		auto y = ply.GetPoints()[i * 3 + 1];
		auto z = ply.GetPoints()[i * 3 + 2];

		VD::AddSphere("Points", { x, y, z }, 0.05f, Color4::White);

		auto code = GetMortonCode(aabbMax, aabbMin, maxDepth, { x, y, z });
		//std::cout << "Morton Code (binary): " << std::bitset<64>(code) << std::endl;

		octants[i].code = code;
		toSort.push_back(code);
	}

	sort(toSort.begin(), toSort.end());

	for (size_t i = 0; i < toSort.size(); i++)
	{
		auto& code = toSort[i];
		//std::cout << "Morton Code (binary): " << std::bitset<64>(code) << std::endl;
	}

	//Plot(pApp, toSort);
	pApp->GetChartRenderer()->DrawOff();

	Octree octree;
	PopulateOctree(&octree, toSort, maxDepth);

	VisualizeOctree(&octree, maxDepth, (aabbMin + aabbMax) * 0.5f, (aabbMax - aabbMin).maxCoeff() * 0.5f);

	//LoadModel(pApp->GetRenderer(), "C:\\Resources\\3D\\PLY\\Complete\\Lower.ply");
	//CUDA::Octree::TestOctree();
}
