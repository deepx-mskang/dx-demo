cmake_minimum_required(VERSION 3.14)
project(yolopv2 DESCRIPTION "YOLOPv2 Demo")

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

add_executable(demo_dxnn_qt5_async
    demo_dxnn_qt5_async.cpp
)

target_compile_definitions(demo_dxnn_qt5_async PRIVATE
    YOLOPV2_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/.."
)

target_include_directories(demo_dxnn_qt5_async PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_compile_options(demo_dxnn_qt5_async PRIVATE -Wall -Wextra -O3)

if(NOT MSVC)
    target_link_options(demo_dxnn_qt5_async PRIVATE -rdynamic)
endif()


target_link_libraries(demo_dxnn_qt5_async
    ${OpenCV_LIBS}
    ${DXRT_LIB}
    Qt5::Widgets
    pthread
)
