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
