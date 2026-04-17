cmake_minimum_required(VERSION 3.14)
project(robotics_demo DESCRIPTION "DEEPX Face Recognition Demo")

add_compile_definitions(PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

if(NOT "${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "${CMAKE_SYSTEM_PROCESSOR}")
    set(CROSS_COMPILE TRUE)
else()
    set(CROSS_COMPILE FALSE)
endif()

set(STD_FS_NO_LIB_NEEDED FALSE)
if(CMAKE_CXX_STANDARD GREATER_EQUAL 17)
    set(STD_FS_NO_LIB_NEEDED TRUE)
endif()

include(dxdemo.cfg)
include(dxdemo.function)

execute_process(
    COMMAND gst-inspect-1.0 vaapidecodebin
    RESULT_VARIABLE GST_VAAPI_FOUND
    OUTPUT_QUIET ERROR_QUIET
)
if(GST_VAAPI_FOUND EQUAL 0)
    add_definitions(-DUSE_VAAPI)
endif()

add_compile_options(-W -Wall -pthread -fstrict-volatile-bitfields -fPIC)
if(ENABLE_DEBUG_INFO)
    add_compile_options(-g -rdynamic)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

add_dxrt_lib()
add_opencv()

add_executable(robotics_demo
    ${CMAKE_SOURCE_DIR}/src/main.cpp
    ${CMAKE_SOURCE_DIR}/src/face_recognition.cpp
    ${CMAKE_SOURCE_DIR}/src/mobed_detector.cpp
    ${CMAKE_SOURCE_DIR}/src/pose_seg.cpp
    ${CMAKE_SOURCE_DIR}/src/ssd.cpp
    ${CMAKE_SOURCE_DIR}/src/nms.cpp
    ${CMAKE_SOURCE_DIR}/src/bbox.cpp
    ${CMAKE_SOURCE_DIR}/src/utils.cpp
)

if(STD_FS_NO_LIB_NEEDED)
    set(STD_FS_LIB "")
else()
    set(STD_FS_LIB stdc++fs)
endif()

target_include_directories(robotics_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(robotics_demo ${link_libs} ${STD_FS_LIB})

install(TARGETS robotics_demo DESTINATION bin)
