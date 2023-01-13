cmake_minimum_required(VERSION 3.25)
project(accelerated_functions)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
# add_library(accelerated_functions AcceleratedFunctions.cpp AcceleratedFunctions.h)
add_subdirectory(extern/pybind11)
#target_include_directories(accelerated_functions PRIVATE "../extern/pybind11//include/pybind11/")
pybind11_add_module(accelerated_functions AcceleratedFunctions.cpp AcceleratedFunctions.h)