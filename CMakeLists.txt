cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(JIT-CL LANGUAGES CXX)

find_package(OpenCL REQUIRED)

add_library(jitcl src/tracer.cpp)
target_include_directories(jitcl PUBLIC include)
target_link_libraries(jitcl PUBLIC OpenCL::OpenCL)

add_executable(Test test.cpp)
target_link_libraries(Test PRIVATE jitcl)