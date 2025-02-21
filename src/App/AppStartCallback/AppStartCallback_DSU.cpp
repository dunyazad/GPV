#include <App/AppStartCallback/AppStartCallback.h>

#include <App/ResourceIO.h>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

unsigned int find(vector<unsigned int>& parents, unsigned int x)
{
	if (parents[x] != x) {
		parents[x] = find(parents, parents[x]);
	}
	return parents[x];
}

void unite(vector<unsigned int>& parents, vector<unsigned int>& ranks, int x, int y) {
	int rootX = find(parents, x);
	int rootY = find(parents, y);
	if (rootX != rootY) {
		if (ranks[rootX] > ranks[rootY]) {
			parents[rootY] = rootX;
		}
		else if (ranks[rootX] < ranks[rootY]) {
			parents[rootX] = rootY;
		}
		else {
			parents[rootY] = rootX;
			ranks[rootX]++;
		}
	}
}

void atomicExch(unsigned int* arr, unsigned int value)
{
	*arr = value;
}

void atomicAdd(unsigned int* arr, unsigned int value)
{
	*arr += value;
}

void unionSets(vector<unsigned int>& parents, vector<unsigned int>& ranks, int x, int y) {
	int rootX = find(parents, x);
	int rootY = find(parents, y);

	if (rootX != rootY) {
		if (ranks[rootX] > ranks[rootY]) {
			atomicExch(&parents[rootY], rootX);
		}
		else if (ranks[rootX] < ranks[rootY]) {
			atomicExch(&parents[rootX], rootY);
		}
		else {
			atomicExch(&parents[rootY], rootX);
			atomicAdd(&ranks[rootX], 1);
		}
	}
}

void mergeClusters(vector<unsigned int>& parents, vector<unsigned int>& ranks, vector<Eigen::Vector3f>& points, int n, float threshold) {
	
	//int idx = threadIdx.x + blockIdx.x * blockDim.x;

#pragma omp parallel for
	for (int idx = 0; idx < points.size(); idx++)
	{
		if (idx < n) {
			for (int j = idx + 1; j < n; j++) {
				float dx = points[idx].x() - points[j].x();
				float dy = points[idx].y() - points[j].y();
				float dz = points[idx].z() - points[j].z();
				float dist = sqrtf(dx * dx + dy * dy + dz * dz);

				if (dist < threshold) {
					unionSets(parents, ranks, idx, j);
				}
			}
		}
	}
}

void AppStartCallback_DSU(App* pApp)
{
	CUDA::DSU::TestDSU();

	return;

	auto t = Time::Now();

	auto renderer = pApp->GetRenderer();

	PLYFormat ply;
	ply.Deserialize(ResourceIO::GetPath("3d/compound.ply").string());

	t = Time::Now();

	vector<Eigen::Vector3f> points;
	vector<unsigned int> parents;
	vector<unsigned int> ranks;

#pragma omp parallel for
	for (size_t i = 0; i < ply.GetPoints().size() / 3; i++)
	{
		auto x = ply.GetPoints()[i * 3];
		auto y = ply.GetPoints()[i * 3 + 1];
		auto z = ply.GetPoints()[i * 3 + 2];

		points.push_back({ x,y,z });
		parents.push_back(0);
		ranks.push_back(1);

		VD::AddSphere("quantized points", { x,y,z }, 0.05f, Color4::White);
	}

	t = Time::End(t, "Loading");

	mergeClusters(parents, ranks, points, points.size(), 0.1f);

	t = Time::End(t, "Clustering");

	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 100.0f, 0.0f, 0.0f }, Color4::Red);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 100.0f, 0.0f }, Color4::Green);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, 100.0f }, Color4::Blue);

	t = Time::End(t, "Visualize");
}
