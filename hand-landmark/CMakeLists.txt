cmake_minimum_required(VERSION 3.14)
project(hand_landmark_deepx_m1_qt5 DESCRIPTION "DEEPX M1 hand detector Qt5 demo")

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

add_executable(hand-landmark
    hand-landmark.cpp
)

add_executable(hand-landmark-detection
    hand-landmark-detection.cpp
)

add_executable(hand-landmark-pose
    hand-landmark-pose.cpp
)

target_include_directories(hand-landmark PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_include_directories(hand-landmark-detection PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_include_directories(hand-landmark-pose PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_compile_options(hand-landmark PRIVATE -Wall -Wextra -O3)
target_compile_options(hand-landmark-detection PRIVATE -Wall -Wextra -O3)
target_compile_options(hand-landmark-pose PRIVATE -Wall -Wextra -O3)

target_link_libraries(hand-landmark PRIVATE
    ${OpenCV_LIBS}
    Qt5::Widgets
    ${DXRT_LIB}
    pthread
)

target_link_libraries(hand-landmark-detection PRIVATE
    ${OpenCV_LIBS}
    Qt5::Widgets
    ${DXRT_LIB}
    pthread
)

target_link_libraries(hand-landmark-pose PRIVATE
    ${OpenCV_LIBS}
    Qt5::Widgets
    ${DXRT_LIB}
    pthread
)
