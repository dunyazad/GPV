cmake_minimum_required(VERSION 3.12)
project(GPV LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# set(CMAKE_VS_PLATFORM_TOOLSET_CUDA "v12.4")

# set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4")  # Adjust this path
# set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/nvcc.exe")  # Optional but might help

find_package(OpenVDB CONFIG REQUIRED)
find_package(VTK REQUIRED)
find_package(CUDAToolkit REQUIRED)  # Use the modern way of finding CUDA

enable_language(CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/bin/Debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/bin/Release)

set(source_list
    src/main.cpp
    src/Common.h
    src/stdHeaderFiles.h
    src/vtkHeaderFiles.h
    src/App/CustomTrackballStyle.h
    src/App/CustomTrackballStyle.cpp
    src/App/Utility.h
    src/App/Utility.cpp
    src/Algorithm/vtkMedianFilter.h
    src/Algorithm/vtkMedianFilter.cpp
    src/Algorithm/vtkQuantizingFilter.h
    src/Algorithm/vtkQuantizingFilter.cpp
    src/Debugging/VisualDebugging.h
    src/Debugging/VisualDebugging.cpp
    src/Debugging/VisualDebuggingLayer.h
    src/Debugging/VisualDebuggingLayer.cpp
    src/Debugging/vtkPolygonalFrustumSource.h
    src/Debugging/vtkPolygonalFrustumSource.cxx
)

add_executable(GPV
    ${source_list}
)

if(MSVC)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT GPV)
    target_compile_options(GPV PRIVATE /bigobj)
endif()

set(VTK_INCLUDE_DIR ${Z_VCPKG_ROOT_DIR}/installed/x64-windows/include/vtk-9.3)
set(VTK_INCLUDE_DIRS ${Z_VCPKG_ROOT_DIR}/installed/x64-windows/include/vtk-9.3)

target_include_directories(GPV PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    "External"
    "External/eigen"
    ${OPENVDB_INCLUDE_DIRS}
    ${VTK_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}  # Add CUDA include directories
)

target_link_libraries(GPV PRIVATE OpenVDB::openvdb)
target_link_libraries(GPV PRIVATE ${VTK_LIBRARIES})
target_link_libraries(GPV PRIVATE CUDA::cudart CUDA::cuda_driver)  # Link CUDA runtime and driver libraries

# Enable CUDA runtime API and any necessary options for your project
set_target_properties(GPV PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON  # Enable separable compilation if needed
    CUDA_STANDARD 11  # Use CUDA 11 (you can adjust this based on your needs)
)

vtk_module_autoinit(
    TARGETS GPV
    MODULES ${VTK_LIBRARIES}
)

function(assign_source_group)
    foreach(_source IN ITEMS ${ARGN})
        if (IS_ABSOLUTE "${_source}")
            file(RELATIVE_PATH _source_rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_source}")
        else()
            set(_source_rel "${_source}")
        endif()
        get_filename_component(_source_path "${_source_rel}" PATH)
        string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
        source_group("${_source_path_msvc}" FILES "${_source}")
    endforeach()
endfunction(assign_source_group)

assign_source_group(${source_list})
