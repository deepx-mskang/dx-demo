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
if(ENABLE_DEBUG_INFO AND NOT CMAKE_BUILD_TYPE MATCHES "^[Rr]elease$|^[Mm]in[Ss]ize[Rr]el$")
    add_compile_options(-g -rdynamic)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

function(strip_release_symbols target_name)
    if(STRIP_RELEASE_BINARIES AND CMAKE_STRIP AND
       CMAKE_BUILD_TYPE MATCHES "^[Rr]elease$|^[Mm]in[Ss]ize[Rr]el$")
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_STRIP} --strip-unneeded $<TARGET_FILE:${target_name}>
            COMMENT "Stripping symbols from $<TARGET_FILE_NAME:${target_name}>"
            VERBATIM
        )
    endif()
endfunction()

add_dxrt_lib()
add_opencv()

find_package(OpenSSL REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets)

add_library(dx_model_crypto SHARED
    ${CMAKE_SOURCE_DIR}/src/encrypted_model_engine.cpp
)
target_compile_definitions(dx_model_crypto PRIVATE DX_MODEL_CRYPTO_BUILD)
target_include_directories(dx_model_crypto PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
set_target_properties(dx_model_crypto PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
target_compile_options(dx_model_crypto PRIVATE
    -fvisibility=hidden
    -fvisibility-inlines-hidden
)
if(NOT MSVC)
    target_link_options(dx_model_crypto PRIVATE
        -Wl,--exclude-libs,ALL
        -Wl,--version-script=${CMAKE_SOURCE_DIR}/cmake/dx_model_crypto.exports
    )
endif()
target_link_libraries(dx_model_crypto PRIVATE
    ${link_libs}
    OpenSSL::Crypto
)
strip_release_symbols(dx_model_crypto)

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
target_link_libraries(robotics_demo Qt5::Widgets ${link_libs} dx_model_crypto ${STD_FS_LIB})
set_target_properties(robotics_demo PROPERTIES
    INSTALL_RPATH "$ORIGIN/../lib"
)
strip_release_symbols(robotics_demo)

install(TARGETS robotics_demo DESTINATION bin)
install(TARGETS dx_model_crypto LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
