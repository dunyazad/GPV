#include <App/Utility.h>

string Miliseconds(const chrono::steady_clock::time_point beginTime, const char* tag)
{
    auto now = chrono::high_resolution_clock::now();
    auto timeSpan = chrono::duration_cast<chrono::nanoseconds>(now - beginTime).count();
    stringstream ss;
    ss << "[[[ ";
    if (nullptr != tag)
    {
        ss << tag << " - ";
    }
    ss << (float)timeSpan / 1000000.0 << " ms ]]]";
    return ss.str();
}

vtkSmartPointer<vtkPolyData> ReadPLY(const std::string& filePath) {
    vtkSmartPointer<vtkPLYReader> reader = vtkSmartPointer<vtkPLYReader>::New();
    reader->SetFileName(filePath.c_str());
    reader->Update();
    return reader->GetOutput();
}

void WritePLY(vtkSmartPointer<vtkPolyData> data, const std::string& filePath) {
    vtkSmartPointer<vtkPLYWriter> writer = vtkSmartPointer<vtkPLYWriter>::New();
    writer->SetFileName(filePath.c_str());
    writer->SetInputData(data);
    writer->Update();
}

#ifdef _WINDOWS
BOOL CALLBACK MonitorEnumProc(HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData) {
    std::vector<MonitorInfo>* monitors = reinterpret_cast<std::vector<MonitorInfo>*>(dwData);
    MONITORINFO monitorInfo;
    monitorInfo.cbSize = sizeof(MONITORINFO);
    if (GetMonitorInfo(hMonitor, &monitorInfo)) {
        monitors->push_back({ hMonitor, monitorInfo });
    }
    return TRUE;
}

void MaximizeConsoleWindowOnMonitor(int monitorIndex) {
    HWND consoleWindow = GetConsoleWindow();
    if (!consoleWindow) return;

    std::vector<MonitorInfo> monitors;
    EnumDisplayMonitors(NULL, NULL, MonitorEnumProc, reinterpret_cast<LPARAM>(&monitors));

    if (monitorIndex >= 0 && monitorIndex < monitors.size()) {
        const MonitorInfo& monitor = monitors[monitorIndex];
        RECT workArea = monitor.monitorInfo.rcWork;

        MoveWindow(consoleWindow, workArea.left, workArea.top,
            workArea.right - workArea.left,
            workArea.bottom - workArea.top, TRUE);
        ShowWindow(consoleWindow, SW_MAXIMIZE);
    }
}

void MaximizeVTKWindowOnMonitor(vtkSmartPointer<vtkRenderWindow> renderWindow, int monitorIndex) {
    std::vector<MonitorInfo> monitors;
    EnumDisplayMonitors(NULL, NULL, MonitorEnumProc, reinterpret_cast<LPARAM>(&monitors));

    if (monitorIndex >= 0 && monitorIndex < monitors.size()) {
        const MonitorInfo& monitor = monitors[monitorIndex];
        RECT workArea = monitor.monitorInfo.rcWork;

        // Set the position and size of the VTK window to match the monitor's work area
        //renderWindow->SetPosition(workArea.left, workArea.top);
        //renderWindow->SetSize(workArea.right - workArea.left, workArea.bottom - workArea.top);

        HWND hwnd = (HWND)renderWindow->GetGenericWindowId();

        MoveWindow(hwnd, workArea.left, workArea.top,
            workArea.right - workArea.left,
            workArea.bottom - workArea.top, TRUE);

        ShowWindow(hwnd, SW_MAXIMIZE);
    }
}
#endif

Eigen::Matrix3f computeRotationMatrix(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
    // Normalize input vectors
    Eigen::Vector3f a_normalized = a.normalized();
    Eigen::Vector3f b_normalized = b.normalized();

    // Compute the cross product and sine of the angle
    Eigen::Vector3f axis = a_normalized.cross(b_normalized);
    double sin_angle = axis.norm();
    double cos_angle = a_normalized.dot(b_normalized);

    // Special case: vectors are parallel
    if (sin_angle < 1e-8) {
        if (cos_angle > 0) {
            // Vectors are aligned
            return Eigen::Matrix3f::Identity();
        }
        else {
            // Vectors are opposite; find a perpendicular vector
            Eigen::Vector3f perp;
            if (std::abs(a_normalized.x()) < std::abs(a_normalized.y()) &&
                std::abs(a_normalized.x()) < std::abs(a_normalized.z())) {
                perp = Eigen::Vector3f(1, 0, 0);
            }
            else if (std::abs(a_normalized.y()) < std::abs(a_normalized.z())) {
                perp = Eigen::Vector3f(0, 1, 0);
            }
            else {
                perp = Eigen::Vector3f(0, 0, 1);
            }
            axis = a_normalized.cross(perp).normalized();
            return Eigen::AngleAxisf(M_PI, axis).toRotationMatrix();
        }
    }

    // Normalize the rotation axis
    axis.normalize();

    // Construct the skew-symmetric matrix for the cross product
    Eigen::Matrix3f K;
    K << 0, -axis.z(), axis.y(),
        axis.z(), 0, -axis.x(),
        -axis.y(), axis.x(), 0;

    // Rodrigues' rotation formula
    Eigen::Matrix3f rotationMatrix = Eigen::Matrix3f::Identity() +
        sin_angle * K +
        (1 - cos_angle) * K * K;

    return rotationMatrix;
}