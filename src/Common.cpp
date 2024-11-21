#include <Common.h>

namespace Time
{
	chrono::steady_clock::time_point Now()
	{
		return chrono::high_resolution_clock::now();
	}

	uint64_t Microseconds(chrono::steady_clock::time_point& from, chrono::steady_clock::time_point& now)
	{
		return std::chrono::duration_cast<std::chrono::microseconds>(now - from).count();
	}

	chrono::steady_clock::time_point End(chrono::steady_clock::time_point& from, const string& message, int number)
	{
		auto now = chrono::high_resolution_clock::now();
		if (-1 == number)
		{
			printf("[%s] %.4f ms from start\n", message.c_str(), (float)(Microseconds(from, now)) / 1000.0f);
		}
		else
		{
			printf("[%6d - %s] %.4f ms from start\n", number, message.c_str(), (float)(Microseconds(from, now)) / 1000.0f);
		}
		return now;
	}

	string DateTime()
	{
		auto t = std::time(nullptr);
		auto tm = *std::localtime(&t);

		std::ostringstream oss;
		oss << std::put_time(&tm, "%Y%m%d_%H%M%S"); // Format: YYYYMMDD_HHMMSS
		return oss.str();
	}
}

Eigen::Vector3f Transform(const Eigen::Matrix4f& tm, const Eigen::Vector3f& p)
{
	return (tm * Eigen::Vector4f(p.x(), p.y(), p.z(), 1.0f)).head<3>();
}

Eigen::Matrix4f vtkToEigen(const vtkMatrix4x4* vtkMat)
{
	Eigen::Matrix4f eigenMat;

	// VTK is row-major, Eigen is column-major by default
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			eigenMat(i, j) = vtkMat->GetElement(i, j);
		}
	}

	return eigenMat;
}

vtkSmartPointer<vtkMatrix4x4> eigenToVtk(const Eigen::Matrix4f& eigenMat)
{
	vtkSmartPointer<vtkMatrix4x4> vtkMat = vtkSmartPointer<vtkMatrix4x4>::New();

	// Eigen is column-major, VTK is row-major
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			vtkMat->SetElement(i, j, eigenMat(i, j));
		}
	}

	return vtkMat;
}
