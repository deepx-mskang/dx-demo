cmake_minimum_required(VERSION 3.14)
project(drone_mixformer DESCRIPTION "MixFormerV2 DEEPX/ONNX tracking demo")

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

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS
        ${ONNXRUNTIME_ROOT}/include
        ${ONNXRUNTIME_ROOT}
        /usr/local/include
        /usr/include
)

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    HINTS
        ${ONNXRUNTIME_ROOT}/lib
        /usr/local/lib
        /usr/lib
        /usr/lib/aarch64-linux-gnu
        /usr/lib/x86_64-linux-gnu
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "ONNX Runtime C++ headers/library not found. Set ONNXRUNTIME_ROOT if installed in a custom path.")
endif()

add_executable(drone_mixformer_qt
    main.cpp
)

target_compile_definitions(drone_mixformer_qt PRIVATE
    DRONE_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_include_directories(drone_mixformer_qt PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

target_compile_options(drone_mixformer PRIVATE -Wall -Wextra -O3)

target_link_libraries(drone_mixformer PRIVATE
    ${OpenCV_LIBS}
    Qt5::Widgets
    ${DXRT_LIB}
    ${ONNXRUNTIME_LIB}
    pthread
    dl
)
