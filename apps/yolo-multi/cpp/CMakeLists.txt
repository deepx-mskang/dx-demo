cmake_minimum_required(VERSION 3.14)
project(yolo_multi_demo DESCRIPTION "DEEPX YOLO Multi-Channel Object Detection Demo")

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

set(DEMO_UTILS_DIR ${CMAKE_SOURCE_DIR}/src/demo_utils)
set(DEMO_UTILS_SRC
    ${DEMO_UTILS_DIR}/yolo_cfg.cpp
    ${DEMO_UTILS_DIR}/yolo.cpp
    ${DEMO_UTILS_DIR}/nms.cpp
    ${DEMO_UTILS_DIR}/bbox.cpp
    ${DEMO_UTILS_DIR}/image.cpp
    ${DEMO_UTILS_DIR}/display.cpp
)

add_executable(yolo_multi_demo
    ${DEMO_UTILS_SRC}
    ${CMAKE_SOURCE_DIR}/src/od.cpp
    ${CMAKE_SOURCE_DIR}/src/yolo_multi_channels.cpp
)

if(STD_FS_NO_LIB_NEEDED)
    set(STD_FS_LIB "")
else()
    set(STD_FS_LIB stdc++fs)
endif()

target_include_directories(yolo_multi_demo PRIVATE
    ${DEMO_UTILS_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/extern/
    ${CMAKE_SOURCE_DIR}/lib/
)
target_link_libraries(yolo_multi_demo ${link_libs} ${STD_FS_LIB})

install(TARGETS yolo_multi_demo DESTINATION bin)
