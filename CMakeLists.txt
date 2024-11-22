
cmake_minimum_required(VERSION 3.5)

# Project name
project(GPV)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/bin/Debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/bin/Release)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

# Find the VTK package
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

#set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};C:\\Program Files (x86)\\hidapi\\lib\\cmake\\hidapi")

# message(STATUS "HDIAPI_INCLUDE_DIRS: ${HDIAPI_INCLUDE_DIRS}")
message(STATUS "HDIAPI_LIBRARIES: ${HDIAPI_LIBRARIES}")

find_package(HDIAPI REQUIRED)
include_directories(${HDIAPI_INCLUDE_DIRS})
link_libraries(${HDIAPI_LIBRARIES})

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
    src/App/USBHandler.h
    src/App/USBHandler.cpp
    src/App/Utility.h
    src/App/Utility.cpp
    src/Algorithm/SVO.h
    src/Algorithm/SVO.cpp
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
    src/AppStartCallback.cpp
    src/main.cpp
)

add_executable(GPV ${source_list})

target_include_directories(GPV PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    "External"
    "External/eigen"
    ${VTK_INCLUDE_DIRS}
)

target_link_libraries(GPV
    ${VTK_LIBRARIES}
    ${HDIAPI_LIBRARIES}
)

if(MSVC)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT GPV)
    target_compile_options(GPV PRIVATE /bigobj)
endif()

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

assign_source_group(${source_list})