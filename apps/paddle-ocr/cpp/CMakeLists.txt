cmake_minimum_required(VERSION 3.14)
project(cam_ppocr_v6_demo DESCRIPTION "DEEPX PP-OCRv6 Qt5 C++ demo")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

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

get_filename_component(CAM_OCR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)

add_executable(cam_ppocr_v6_demo
    main.cpp
    ocr_engine.cpp
    ocr_engine.hpp
)

target_compile_definitions(cam_ppocr_v6_demo PRIVATE
    CAM_OCR_ROOT_DIR="${CAM_OCR_ROOT}"
)

target_include_directories(cam_ppocr_v6_demo PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_compile_options(cam_ppocr_v6_demo PRIVATE -Wall -Wextra -O3)

target_link_libraries(cam_ppocr_v6_demo PRIVATE
    ${OpenCV_LIBS}
    Qt5::Widgets
    ${DXRT_LIB}
    pthread
)
