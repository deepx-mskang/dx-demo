cmake_minimum_required(VERSION 3.14)
project(yolo26s_3_qt DESCRIPTION "YOLO26-S 3-task Qt demo")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets)

if(CROSS_COMPILE OR MSVC)
    find_library(DXRT_LIB dxrt HINTS ${DXRT_INSTALLED_DIR}/lib REQUIRED)
    include_directories(${DXRT_INSTALLED_DIR}/include)
    link_directories(${DXRT_INSTALLED_DIR}/lib)
else()
    find_package(dxrt REQUIRED HINTS ${DXRT_INSTALLED_DIR})
    set(DXRT_LIB dxrt)
endif()

set(OD_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../object_detection/yolo26s")
set(SEG_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../instance_segmentation/yolo26s_seg")
set(POSE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../pose_estimation/yolo26s_pose")

if(NOT DEFINED PROJECT_ROOT_OVERRIDE)
    get_filename_component(PROJECT_ROOT_OVERRIDE "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
endif()

add_executable(yolo26s_3
    yolo26s_3.cpp
    "${OD_ROOT}/utility/common_util.cpp"
)

target_compile_definitions(yolo26s_3 PRIVATE PROJECT_ROOT_DIR="${PROJECT_ROOT_OVERRIDE}/yolo26")

target_include_directories(yolo26s_3 PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    "${OD_ROOT}"
    "${OD_ROOT}/utility"
    "${OD_ROOT}/extern"
    "${SEG_ROOT}"
    "${POSE_ROOT}"
)

target_compile_options(yolo26s_3 PRIVATE -Wall -Wextra -O3)

target_link_libraries(yolo26s_3
    Qt5::Widgets
    ${OpenCV_LIBS}
    ${DXRT_LIB}
)
