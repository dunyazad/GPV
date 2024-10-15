cmake_minimum_required(VERSION 3.12)
project(GPV LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/wd4819")

# Find necessary packages
find_package(OpenVDB CONFIG REQUIRED)
find_package(VTK REQUIRED)
find_package(CUDAToolkit REQUIRED)

enable_language(CUDA)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/bin/Debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/bin/Release)

# List of CUDA source files
set(cuda_source_list
    src/CUDA/KDTree.cuh
    src/CUDA/KDTree.cu
)

# List of C++ source files
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

# Set CUDA source file properties
set_source_files_properties(${cuda_source_list} PROPERTIES LANGUAGE CUDA)

# Compile CUDA files into an object library
add_library(cuda_objects OBJECT ${cuda_source_list})

# Create the final executable by linking C++ sources and CUDA object files
add_executable(GPV
    ${source_list}
    $<TARGET_OBJECTS:cuda_objects>  # Link CUDA object files here
)

# Optional: Set MSVC-specific options
if(MSVC)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT GPV)
    target_compile_options(GPV PRIVATE /bigobj)
endif()

target_include_directories(GPV PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    "External"
    "External/eigen"
    ${OPENVDB_INCLUDE_DIRS}
    ${VTK_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_include_directories(cuda_objects PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    "External"
    "External/eigen"
    ${OPENVDB_INCLUDE_DIRS}
    ${VTK_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(GPV PRIVATE OpenVDB::openvdb)
target_link_libraries(GPV PRIVATE ${VTK_LIBRARIES})
target_link_libraries(GPV PRIVATE CUDA::cudart CUDA::cuda_driver)

# Enable separable compilation if needed, or disable for faster builds
set_target_properties(GPV PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF  # Change to ON if needed
    CUDA_STANDARD 11  # Adjust based on your CUDA version
)

vtk_module_autoinit(
    TARGETS GPV
    MODULES ${VTK_LIBRARIES}
)

# Group source files in IDE
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

assign_source_group(${cuda_source_list})
assign_source_group(${source_list})
