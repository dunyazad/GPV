#include <App/AppStartCallback/AppStartCallback.h>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

void AppStartCallback_ShrinkPatch(App* pApp)
{
	auto renderer = pApp->GetRenderer();

	auto t = Time::Now();

	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 100.0f, 0.0f, 0.0f }, Color4::Red);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 100.0f, 0.0f }, Color4::Green);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, 100.0f }, Color4::Blue);

	PLYFormat ply;
	ply.Deserialize("C:\\Resources\\3D\\PLY\\Complete\\sourcePatch.ply");

	float voxelSize = 0.1f;
	auto aabbMin = ply.GetAABBMin();
	auto aabbMax = ply.GetAABBMax();

	auto volumeMinX = floorf(aabbMin.x() / voxelSize) * voxelSize;
	auto volumeMinY = floorf(aabbMin.y() / voxelSize) * voxelSize;
	auto volumeMinZ = floorf(aabbMin.z() / voxelSize) * voxelSize;
	auto volumeMaxX = ceilf(aabbMax.x() / voxelSize) * voxelSize;
	auto volumeMaxY = ceilf(aabbMax.y() / voxelSize) * voxelSize;
	auto volumeMaxZ = ceilf(aabbMax.z() / voxelSize) * voxelSize;

	auto volumeMin = Eigen::Vector3f(volumeMinX, volumeMinY, volumeMinZ);
	auto volumeMax = Eigen::Vector3f(volumeMaxX, volumeMaxY, volumeMaxZ);

	Eigen::Vector3f dimensions = volumeMax - volumeMin;

	unsigned int countX = (unsigned int)ceilf(dimensions.x() / voxelSize);
	unsigned int countY = (unsigned int)ceilf(dimensions.y() / voxelSize);
	unsigned int countZ = 10;

	unsigned int* grid = new unsigned int[countX * countY * countZ];
	for (size_t i = 0; i < countX * countY * countZ; i++)
	{
		grid[i] = UINT32_MAX;
	}

	for (size_t i = 0; i < ply.GetPoints().size() / 3; i++)
	{
		auto x = ply.GetPoints()[i * 3];
		auto y = ply.GetPoints()[i * 3 + 1];
		auto z = ply.GetPoints()[i * 3 + 2];

		auto r = ply.GetColors()[i * 3];
		auto g = ply.GetColors()[i * 3 + 1];
		auto b = ply.GetColors()[i * 3 + 2];

		VD::AddCube("cubes", { x,y,z }, 0.05f, Color4::FromNormalized(r, g, b, 1.0f));

		unsigned int xIndex = (unsigned int)floorf((x - volumeMin.x()) / voxelSize);
		unsigned int yIndex = (unsigned int)floorf((y - volumeMin.y()) / voxelSize);

		unsigned int zIndex = 1;
		while (UINT32_MAX != grid[zIndex * countX * countY + yIndex * countX + xIndex])
		{
			zIndex++;
		}

		//if (zIndex > 1) printf("zIndex : %d\n", zIndex);

		grid[zIndex * countX * countY + yIndex * countX + xIndex] = i;
	}

	for (size_t yIndex = 0; yIndex < countY; yIndex++)
	{
		for (size_t xIndex = 0; xIndex < countX; xIndex++)
		{
			//size_t zIndex = 1;
			//if (UINT32_MAX != grid[zIndex * countX * countY + yIndex * countX + xIndex])
			//{
			//	float x = (float)xIndex * voxelSize + volumeMin.x() + voxelSize * 0.5f;
			//	float y = (float)yIndex * voxelSize + volumeMin.y() + voxelSize * 0.5f;

			//	VD::AddCube("cubes", { x,y, 0.0f }, 0.05f, Color4::Red);
			//}

			if (UINT32_MAX == grid[countX * countY + yIndex * countX + xIndex]) continue;

			//{
			//	float x = (float)xIndex * voxelSize + volumeMin.x() + voxelSize * 0.5f;
			//	float y = (float)yIndex * voxelSize + volumeMin.y() + voxelSize * 0.5f;

			//	VD::AddCube("cubes", { x,y, 0.0f }, 0.05f, Color4::Red);
			//}

			unsigned int neighborCount = 0;

			for (int yOffset = -1; yOffset <= 1; yOffset++)
			{
				if (0 == yIndex || countY == yIndex) continue;

				for (int xOffset = -1; xOffset <= 1; xOffset++)
				{
					if (0 == xIndex || countX == xIndex) continue;

					if (0 == xIndex + xOffset && 0 == yIndex + yOffset)continue;

					if (UINT32_MAX != grid[countX * countY + (yIndex + yOffset) * countX + xIndex + xOffset])
					{
						neighborCount++;
					}
				}
			}

			printf("neighborCount : %d\n", neighborCount);

			grid[yIndex * countX + xIndex] = neighborCount;

			if (neighborCount < 3)
			{
				float x = (float)xIndex * voxelSize + volumeMin.x() + voxelSize * 0.5f;
				float y = (float)yIndex * voxelSize + volumeMin.y() + voxelSize * 0.5f;

				VD::AddCube("cubes", { x,y, 0.0f }, 0.05f, Color4::Red);
			}
		}
	}

	t = Time::End(t, "Visualize");
}
