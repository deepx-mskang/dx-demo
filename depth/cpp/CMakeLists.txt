cmake_minimum_required(VERSION 3.14)
project(cpp_depth_async DESCRIPTION "Depth Anything async C++ demo")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

if(CROSS_COMPILE OR MSVC)
    find_library(DXRT_LIB dxrt HINTS ${DXRT_INSTALLED_DIR}/lib REQUIRED)
    include_directories(${DXRT_INSTALLED_DIR}/include)
    link_directories(${DXRT_INSTALLED_DIR}/lib)
else()
    find_package(dxrt REQUIRED HINTS ${DXRT_INSTALLED_DIR})
    set(DXRT_LIB dxrt)
endif()

find_package(X11 QUIET)

if(NOT DEFINED DEPTH_ROOT_OVERRIDE)
    get_filename_component(DEPTH_ROOT_OVERRIDE "${CMAKE_CURRENT_SOURCE_DIR}/../../depth" ABSOLUTE)
endif()

add_executable(cpp_depth
    main.cpp
)

add_executable(camera_test
    test.cpp
)

target_compile_definitions(cpp_depth PRIVATE DEPTH_ROOT_DIR="${DEPTH_ROOT_OVERRIDE}")

if(X11_FOUND)
    target_compile_definitions(cpp_depth PRIVATE USE_X11)
endif()

target_include_directories(cpp_depth PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_compile_options(cpp_depth PRIVATE -Wall -Wextra -O3)
target_compile_options(camera_test PRIVATE -Wall -Wextra -O3)

if(NOT MSVC)
    target_link_options(cpp_depth PRIVATE -rdynamic)
endif()

target_link_libraries(cpp_depth
    ${OpenCV_LIBS}
    ${DXRT_LIB}
    pthread
)

target_link_libraries(camera_test
    ${OpenCV_LIBS}
)

if(X11_FOUND)
    target_link_libraries(cpp_depth ${X11_LIBRARIES})
endif()
