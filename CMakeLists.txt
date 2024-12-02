
cmake_minimum_required(VERSION 3.5)

# Project name
project(GPV LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/wd4819")

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/bin/Debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/bin/Release)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

# Find the VTK package
#find_package(OpenVDB CONFIG REQUIRED)
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})
find_package(CUDAToolkit REQUIRED)

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};C:\\Program Files (x86)\\hidapi\\lib\\cmake\\hidapi")

enable_language(CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75 86) # Replace with architectures suitable for your GPU(s)

IF (WIN32)
# message(STATUS "HDIAPI_INCLUDE_DIRS: ${HDIAPI_INCLUDE_DIRS}")
# message(STATUS "HDIAPI_LIBRARIES: ${HDIAPI_LIBRARIES}")

find_package(HDIAPI REQUIRED)
include_directories(
${HDIAPI_INCLUDE_DIRS}
#"C:\\Program Files (x86)\\OpenVDB\\include"
)
link_libraries(
${HDIAPI_LIBRARIES}
#"C:\\Program Files (x86)\\OpenVDB\\lib\\openvdb.lib"
)
ENDIF()

set(cuda_source_list
    src/CUDA/CUDA_Common.cuh
    src/CUDA/CUDA.cuh
    src/CUDA/CUDA.cu
    src/CUDA/Octree.cuh
    src/CUDA/Octree.cu
    src/CUDA/RegularGrid.cuh
    src/CUDA/RegularGrid.cu
)

set(source_list
    src/stdHeaderFiles.h
    src/vtkHeaderFiles.h
    src/Common.h
    src/Common.cpp
    src/Color.h
    src/Color.cpp
    src/Max.h
    src/Max.cpp
    src/App/App.h
    src/App/App.cpp
    src/App/AppEventHandlers.h
    src/App/AppEventHandlers.cpp
    src/App/CustomTrackballStyle.h
    src/App/CustomTrackballStyle.cpp
    src/App/Serialization.hpp
    src/App/AppStartCallback.h
    src/App/AppStartCallback.cpp
    src/App/AppStartCallback_Capture.cpp
    src/App/AppStartCallback_Integrate.cpp
    src/App/AppStartCallback_Octree.cpp
    src/App/AppStartCallback_RegularGrid.cpp
    src/App/AppStartCallback_Simple.cpp
    src/App/USBHandler.h
    src/App/USBHandler.cpp
    src/App/Utility.h
    src/App/Utility.cpp
    src/Algorithm/SVO.h
    src/Algorithm/SVO.cpp
    src/Algorithm/MiniMath.h
    src/Algorithm/MiniMath.cpp
    src/Algorithm/HVETM.h
    src/Algorithm/HVETM.cpp
    src/Algorithm/KDtree.h
    src/Algorithm/KDTree.cpp
    src/Algorithm/Octree.hpp
    src/Algorithm/CustomPolyDataFilter.h
    src/Algorithm/CustomPolyDataFilter.cpp
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
    src/main.cpp
)

set_source_files_properties(${cuda_source_list} PROPERTIES LANGUAGE CUDA)

add_library(cuda_objects OBJECT ${cuda_source_list})

add_executable(GPV
 ${source_list}
$<TARGET_OBJECTS:cuda_objects>  # Link CUDA object files here
)

target_include_directories(GPV PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    "External"
    "External/eigen"
    #"External/nanoflann"
    #"External/nanovdb"
    ${VTK_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_include_directories(cuda_objects PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    "External"
    "External/eigen"
    #"External/nanoflann"
    #"External/nanovdb"
    ${VTK_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(GPV
    ${VTK_LIBRARIES}
    ${HDIAPI_LIBRARIES}
    CUDA::cudart CUDA::cuda_driver
    CUDA::cusolver CUDA::cusparse
)

target_link_libraries(cuda_objects PRIVATE
    ${VTK_LIBRARIES}
 )

if(MSVC)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT GPV)
    target_compile_options(GPV PRIVATE /bigobj)
endif()

set_target_properties(GPV PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON  # Change to ON if needed
    CMAKE_CUDA_ARCHITECTURES "60;70;80"
    CUDA_STANDARD 11  # Adjust based on your CUDA version
)

# vtk_module_autoinit(
#     TARGETS GPV
#     MODULES ${VTK_LIBRARIES}
# )

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