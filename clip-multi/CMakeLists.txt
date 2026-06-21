cmake_minimum_required(VERSION 3.14)
project(clip_multi_cpp DESCRIPTION "DEEPX CLIP Qt5 multi-stream demo")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(OpenCV REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED IMPORTED_TARGET
    gstreamer-1.0
    gstreamer-app-1.0
    gstreamer-video-1.0
)

add_executable(video_decode_preview video_decode_preview.cpp)

target_compile_definitions(video_decode_preview PRIVATE
    CLIP_MULTI_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_include_directories(video_decode_preview PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
)

target_compile_options(video_decode_preview PRIVATE -Wall -Wextra -O3)

target_link_libraries(video_decode_preview PRIVATE
    ${OpenCV_LIBS}
    Qt5::Widgets
    PkgConfig::GSTREAMER
    pthread
)

option(BUILD_CLIP_MULTI_CPP "Build the CLIP inference application" ON)

if(BUILD_CLIP_MULTI_CPP)
    find_package(ZLIB REQUIRED)

    if(CROSS_COMPILE OR MSVC)
        find_library(DXRT_LIB dxrt HINTS ${DXRT_INSTALLED_DIR}/lib REQUIRED)
        set(DXRT_INCLUDE_DIR ${DXRT_INSTALLED_DIR}/include)
    else()
        find_package(dxrt REQUIRED HINTS ${DXRT_INSTALLED_DIR})
        set(DXRT_LIB dxrt)
    endif()

    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        HINTS ${ONNXRUNTIME_ROOT}/include ${ONNXRUNTIME_ROOT} /usr/local/include /usr/include
        REQUIRED
    )
    find_library(ONNXRUNTIME_LIB
        NAMES onnxruntime
        HINTS ${ONNXRUNTIME_ROOT}/lib /usr/local/lib /usr/lib
        REQUIRED
    )

    add_executable(clip_multi_cpp main.cpp clip_tokenizer.cpp clip_tokenizer.hpp)

    target_compile_definitions(clip_multi_cpp PRIVATE
        CLIP_MULTI_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )

    target_include_directories(clip_multi_cpp PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${ONNXRUNTIME_INCLUDE_DIR}
        ${DXRT_INCLUDE_DIR}
        ${GSTREAMER_INCLUDE_DIRS}
    )

    target_compile_options(clip_multi_cpp PRIVATE -Wall -Wextra -O3)

    target_link_libraries(clip_multi_cpp PRIVATE
        ${OpenCV_LIBS}
        Qt5::Widgets
        ZLIB::ZLIB
        ${DXRT_LIB}
        ${ONNXRUNTIME_LIB}
        PkgConfig::GSTREAMER
        pthread
        dl
    )
endif()
