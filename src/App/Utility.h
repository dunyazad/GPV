#pragma once

#include <Common.h>

string Miliseconds(const chrono::steady_clock::time_point beginTime, const char* tag = nullptr);
#define TS(name) auto time_##name = chrono::high_resolution_clock::now()();
#define TE(name) std::cout << Miliseconds(time_##name, #name) << std::endl;

vtkSmartPointer<vtkPolyData> ReadPLY(const std::string& filePath);
void WritePLY(vtkSmartPointer<vtkPolyData> data, const std::string& filePath);

struct MonitorInfo {
    HMONITOR hMonitor;
    MONITORINFO monitorInfo;
};

BOOL CALLBACK MonitorEnumProc(HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData);
void MaximizeConsoleWindowOnMonitor(int monitorIndex);
void MaximizeVTKWindowOnMonitor(vtkSmartPointer<vtkRenderWindow> renderWindow, int monitorIndex);