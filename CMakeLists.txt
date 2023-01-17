cmake_minimum_required(VERSION 3.25)
project(precompiled_functions)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
add_subdirectory(extern/pybind11)
pybind11_add_module(precompiled_functions PrecompiledFunctions.cpp)
target_include_directories(precompiled_functions
 PRIVATE
  extern/eigen)