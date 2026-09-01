cmake_minimum_required(VERSION 3.14)
project(serial_ocr_server DESCRIPTION "DEEPX Serial-QR web demo server (PP-OCRv6 + MJPEG)")

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

get_filename_component(SERIAL_QR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
get_filename_component(PADDLE_OCR_CPP_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../paddle-ocr/cpp" ABSOLUTE)

if(NOT EXISTS "${PADDLE_OCR_CPP_DIR}/ocr_engine.cpp")
    message(FATAL_ERROR
        "apps/paddle-ocr/cpp/ocr_engine.cpp 를 찾을 수 없습니다: ${PADDLE_OCR_CPP_DIR}")
endif()

# OCR 엔진은 apps/paddle-ocr 의 소스를 그대로 컴파일해 재사용한다 (복제 없음).
# ocr_engine.cpp 는 Qt 의존이 없으므로 GUI 없이 링크된다.
add_executable(serial_ocr_server
    serial_ocr_server.cpp
    device_registry.cpp
    ${PADDLE_OCR_CPP_DIR}/ocr_engine.cpp
)

target_compile_definitions(serial_ocr_server PRIVATE
    SERIAL_QR_ROOT_DIR="${SERIAL_QR_ROOT}"
    CAM_OCR_ROOT_DIR="${PADDLE_OCR_CPP_DIR}"
)

target_include_directories(serial_ocr_server PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${PADDLE_OCR_CPP_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_compile_options(serial_ocr_server PRIVATE -Wall -Wextra -O3)

find_package(Threads REQUIRED)

target_link_libraries(serial_ocr_server PRIVATE
    ${OpenCV_LIBS}
    ${DXRT_LIB}
    Threads::Threads
)
