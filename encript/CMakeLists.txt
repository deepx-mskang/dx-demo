cmake_minimum_required(VERSION 3.16)

project(dxnn_encript_tools LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenSSL REQUIRED)

add_executable(enc enc.cpp)
target_link_libraries(enc PRIVATE OpenSSL::Crypto)

add_executable(dec dec.cpp)
target_link_libraries(dec PRIVATE OpenSSL::Crypto)

