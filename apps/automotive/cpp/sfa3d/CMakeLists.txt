cmake_minimum_required(VERSION 3.16)

project(SFA3D_DEEPX_CPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(DXRT_ROOT "/home/donggyun/git/dx-all-suite/dx-runtime/dx_rt" CACHE PATH "Path to dx_rt")

find_package(OpenCV REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets)

find_library(DXRT_LIBRARY
    NAMES dxrt
    HINTS
        "${DXRT_ROOT}/build_x86_64/lib"
        "${DXRT_ROOT}/lib"
        "${DXRT_ROOT}/lib64"
)

if(NOT DXRT_LIBRARY)
    message(FATAL_ERROR "Could not find libdxrt. Set -DDXRT_ROOT=/path/to/dx_rt")
endif()

add_executable(sfa3d_async
    sfa/demo_dxnn_async.cpp
)

target_include_directories(sfa3d_async PRIVATE
    "${DXRT_ROOT}/lib/include"
    "${DXRT_ROOT}/extern/include"
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(sfa3d_async PRIVATE
    ${DXRT_LIBRARY}
    ${OpenCV_LIBS}
    Qt5::Widgets
    pthread
)

target_compile_definitions(sfa3d_async PRIVATE
    SFA3D_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)

get_filename_component(DXRT_LIBRARY_DIR "${DXRT_LIBRARY}" DIRECTORY)
set_target_properties(sfa3d_async PROPERTIES
    BUILD_RPATH "${DXRT_LIBRARY_DIR}"
)
